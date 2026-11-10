#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

class ThreadPool
{
public:
    explicit ThreadPool(std::size_t threadCount, std::size_t spinCount = 2048);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    [[nodiscard]]
    std::size_t ThreadCount() const noexcept;

    template <class Function>
    void ParallelFor(std::size_t begin, std::size_t end, Function&& function, std::size_t grainSize = 1);

private:
    static constexpr std::size_t CacheLineSize = 64;
    static constexpr std::size_t InlineTaskSize = 64;
    static constexpr std::size_t InvalidWorkerIndex = static_cast<std::size_t>(-1);

    class Task
    {
    public:
        Task() = default;

        template <class Function>
            requires (!std::is_same_v<std::remove_cvref_t<Function>, Task>)
        explicit Task(Function&& function)
        {
            Emplace(std::forward<Function>(function));
        }

        ~Task()
        {
            Reset();
        }

        Task(const Task&) = delete;
        Task& operator=(const Task&) = delete;

        Task(Task&& other) noexcept
        {
            MoveFrom(std::move(other));
        }

        Task& operator=(Task&& other) noexcept
        {
            if (this != &other)
            {
                Reset();
                MoveFrom(std::move(other));
            }

            return *this;
        }

        void operator()() const
        {
            m_pInvokeFunc(m_pObject);
        }

        [[nodiscard]]
        explicit operator bool() const noexcept
        {
            return m_pObject != nullptr;
        }

    private:
        using InvokeFunction = void (*)(void*);
        using MoveFunction = void (*)(void*, void*);
        using DestroyFunction = void (*)(void*);

        template <class Function>
        void Emplace(Function&& function)
        {
            using FunctionType = std::decay_t<Function>;

            constexpr bool useInlineStorage = sizeof(FunctionType) <= InlineTaskSize &&
                alignof(FunctionType) <= alignof(std::max_align_t) &&
                std::is_nothrow_move_constructible_v<FunctionType>;

            m_pInvokeFunc = [](void* pObj)
            {
                (*static_cast<FunctionType*>(pObj))();
            };

            if constexpr (useInlineStorage)
            {
                m_pObject = ::new(static_cast<void*>(m_Storage)) FunctionType(std::forward<Function>(function));

                m_pMoveFunc = [](void* pSrc, void* pDest)
                {
                    auto* pSrcFunc = static_cast<FunctionType*>(pSrc);

                    ::new(pDest) FunctionType(std::move(*pSrcFunc));

                    pSrcFunc->~FunctionType();
                };

                m_pDestroyFunc = [](void* pObj)
                {
                    static_cast<FunctionType*>(pObj)->~FunctionType();
                };

                m_HeapAllocated = false;
            }
            else
            {
                m_pObject = new FunctionType(std::forward<Function>(function));

                m_pMoveFunc = nullptr;

                m_pDestroyFunc = [](void* pObj)
                {
                    delete static_cast<FunctionType*>(pObj);
                };

                m_HeapAllocated = true;
            }
        }

        void Reset() noexcept
        {
            if (m_pObject != nullptr)
                m_pDestroyFunc(m_pObject);

            m_pObject = nullptr;
            m_pInvokeFunc = nullptr;
            m_pMoveFunc = nullptr;
            m_pDestroyFunc = nullptr;
            m_HeapAllocated = false;
        }

        void MoveFrom(Task&& other) noexcept
        {
            if (!other)
                return;

            m_pInvokeFunc = other.m_pInvokeFunc;
            m_pMoveFunc = other.m_pMoveFunc;
            m_pDestroyFunc = other.m_pDestroyFunc;
            m_HeapAllocated = other.m_HeapAllocated;

            if (other.m_HeapAllocated)
            {
                m_pObject = other.m_pObject;
            }
            else
            {
                m_pMoveFunc(other.m_pObject, static_cast<void*>(m_Storage));

                m_pObject = static_cast<void*>(m_Storage);
            }

            other.m_pObject = nullptr;
            other.m_pInvokeFunc = nullptr;
            other.m_pMoveFunc = nullptr;
            other.m_pDestroyFunc = nullptr;
            other.m_HeapAllocated = false;
        }

        alignas(std::max_align_t) std::byte m_Storage[InlineTaskSize]{};

        void* m_pObject{};

        InvokeFunction m_pInvokeFunc{};
        MoveFunction m_pMoveFunc{};
        DestroyFunction m_pDestroyFunc{};

        bool m_HeapAllocated{};
    };

    struct alignas(CacheLineSize) WorkerQueue
    {
        std::mutex mutex;
        std::deque<Task> tasks;
    };

    struct CompletionState
    {
        explicit CompletionState(std::size_t taskCount) : remaining(taskCount)
        {
        }

        std::atomic<std::size_t> remaining;

        std::mutex exceptionMutex;
        std::exception_ptr exception;

        void Complete(std::size_t count = 1) noexcept
        {
            if (remaining.fetch_sub(count, std::memory_order_acq_rel) == count)
                remaining.notify_one();
        }

        void CaptureException(const std::exception_ptr& currentException)
        {
            std::lock_guard lock(exceptionMutex);

            if (!exception)
                exception = currentException;
        }

        [[nodiscard]]
        std::exception_ptr Exception()
        {
            std::lock_guard lock(exceptionMutex);

            return exception;
        }
    };

    class ParallelRegionGuard
    {
    public:
        ParallelRegionGuard()
        {
            ++s_ParallelDepth;
        }

        ~ParallelRegionGuard()
        {
            --s_ParallelDepth;
        }

        ParallelRegionGuard(const ParallelRegionGuard&) = delete;
        ParallelRegionGuard& operator=(const ParallelRegionGuard&) = delete;
    };

    static void CpuRelax() noexcept;

    void WorkerLoop(std::size_t workerIndex);
    void Enqueue(Task task);

    [[nodiscard]]
    bool TryExecuteOne(std::size_t preferredWorkerIndex);

    [[nodiscard]]
    bool TryPopLocal(std::size_t workerIndex, Task& task);

    [[nodiscard]]
    bool TrySteal(std::size_t thiefWorkerIndex, Task& task);

    void WaitForCompletion(CompletionState& state);

    std::size_t m_ThreadCount;
    std::size_t m_WorkerCount;
    std::size_t m_SpinCount;

    std::vector<std::thread> m_Workers;
    std::vector<std::unique_ptr<WorkerQueue>> m_Queues;

    alignas(CacheLineSize) std::atomic<std::size_t> m_QueuedTaskCount{};
    alignas(CacheLineSize) std::atomic<std::size_t> m_NextQueue{};
    alignas(CacheLineSize) std::atomic<std::uint64_t> m_WorkEpoch{};
    alignas(CacheLineSize) std::atomic<bool> m_Stop{};

    inline static thread_local ThreadPool* s_pCurrentPool{};
    inline static thread_local std::size_t s_CurrentWorkerIndex{InvalidWorkerIndex};
    inline static thread_local std::size_t s_ParallelDepth{};
};

template <class Function>
void ThreadPool::ParallelFor(std::size_t begin, std::size_t end, Function&& function, std::size_t grainSize)
{
    if (begin >= end)
        return;

    if (grainSize == 0)
        grainSize = 1;

    const auto workSize = end - begin;

    if (m_ThreadCount == 1 || workSize <= grainSize || s_pCurrentPool == this || s_ParallelDepth != 0)
    {
        function(begin, end);
        return;
    }

    const auto chunkFromThreads = workSize / m_ThreadCount + static_cast<std::size_t>(workSize % m_ThreadCount != 0);
    const auto chunkSize = std::max(grainSize, chunkFromThreads);
    const auto taskCount = workSize / chunkSize + static_cast<std::size_t>(workSize % chunkSize != 0);

    if (taskCount == 1)
    {
        function(begin, end);
        return;
    }

    CompletionState completion(taskCount - 1);
    ParallelRegionGuard parallelRegion;

    for (std::size_t taskIndex = 1; taskIndex < taskCount; ++taskIndex)
    {
        const auto chunkBegin = begin + taskIndex * chunkSize;
        const auto chunkEnd = chunkBegin + std::min(chunkSize, end - chunkBegin);

        try
        {
            Enqueue(Task([&, chunkBegin, chunkEnd]
            {
                try
                {
                    function(chunkBegin, chunkEnd);
                }
                catch (...)
                {
                    completion.CaptureException(std::current_exception());
                }

                completion.Complete();
            }));
        }
        catch (...)
        {
            completion.CaptureException(std::current_exception());
            completion.Complete(taskCount - taskIndex);

            break;
        }
    }

    const auto callerChunkEnd = begin + std::min(chunkSize, workSize);

    try
    {
        function(begin, callerChunkEnd);
    }
    catch (...)
    {
        completion.CaptureException(std::current_exception());
    }

    WaitForCompletion(completion);

    if (const auto& exception = completion.Exception())
        std::rethrow_exception(exception);
}
