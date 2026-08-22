#include "ThreadPool.hpp"

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

ThreadPool::ThreadPool(std::size_t threadCount, std::size_t spinCount) :
    m_ThreadCount(std::max<std::size_t>(threadCount, 1)), m_WorkerCount(m_ThreadCount - 1), m_SpinCount(spinCount)
{
    m_Queues.reserve(m_WorkerCount);
    m_Workers.reserve(m_WorkerCount);

    for (std::size_t workerIndex{}; workerIndex < m_WorkerCount; ++workerIndex)
        m_Queues.emplace_back(std::make_unique<WorkerQueue>());

    for (std::size_t workerIndex{}; workerIndex < m_WorkerCount; ++workerIndex)
        m_Workers.emplace_back([this, workerIndex] { WorkerLoop(workerIndex); });
}

ThreadPool::~ThreadPool()
{
    m_Stop.store(true, std::memory_order_release);

    m_WorkEpoch.fetch_add(1, std::memory_order_release);
    m_WorkEpoch.notify_all();

    for (auto& worker : m_Workers)
        worker.join();
}

std::size_t ThreadPool::ThreadCount() const noexcept
{
    return m_ThreadCount;
}

void ThreadPool::CpuRelax() noexcept
{
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

void ThreadPool::WorkerLoop(std::size_t workerIndex)
{
    s_CurrentPool = this;
    s_CurrentWorkerIndex = workerIndex;

    while (true)
    {
        if (TryExecuteOne(workerIndex))
            continue;

        bool foundWork = false;

        for (std::size_t spin{}; spin < m_SpinCount; ++spin)
        {
            if (TryExecuteOne(workerIndex))
            {
                foundWork = true;
                break;
            }

            CpuRelax();
        }

        if (foundWork)
            continue;

        if (m_Stop.load(std::memory_order_acquire))
        {
            if (m_QueuedTaskCount.load(std::memory_order_acquire) == 0)
                break;

            CpuRelax();

            continue;
        }

        const auto observedEpoch = m_WorkEpoch.load(std::memory_order_acquire);

        if (TryExecuteOne(workerIndex))
            continue;

        if (m_Stop.load(std::memory_order_acquire))
            continue;

        m_WorkEpoch.wait(observedEpoch, std::memory_order_acquire);
    }

    s_CurrentWorkerIndex = InvalidWorkerIndex;
    s_CurrentPool = nullptr;
}

void ThreadPool::Enqueue(Task task)
{
    if (m_WorkerCount == 0)
    {
        task();

        return;
    }

    if (m_Stop.load(std::memory_order_acquire))
        throw std::runtime_error("ThreadPool is stopping");

    std::size_t queueIndex{};

    if (s_CurrentPool == this && s_CurrentWorkerIndex < m_WorkerCount)
        queueIndex = s_CurrentWorkerIndex;
    else
        queueIndex = m_NextQueue.fetch_add(1, std::memory_order_relaxed) % m_WorkerCount;

    auto& queue = *m_Queues[queueIndex];

    {
        std::lock_guard lock(queue.mutex);

        queue.tasks.emplace_back(std::move(task));

        m_QueuedTaskCount.fetch_add(1, std::memory_order_release);
    }

    m_WorkEpoch.fetch_add(1, std::memory_order_release);
    m_WorkEpoch.notify_one();
}

bool ThreadPool::TryExecuteOne(std::size_t preferredWorkerIndex)
{
    Task task;

    if (preferredWorkerIndex < m_WorkerCount && TryPopLocal(preferredWorkerIndex, task))
    {
        task();

        return true;
    }

    if (TrySteal(preferredWorkerIndex, task))
    {
        task();

        return true;
    }

    return false;
}

bool ThreadPool::TryPopLocal(std::size_t workerIndex, Task& task)
{
    auto& queue = *m_Queues[workerIndex];

    std::lock_guard lock(queue.mutex);

    if (queue.tasks.empty())
        return false;

    task = std::move(queue.tasks.back());

    queue.tasks.pop_back();

    m_QueuedTaskCount.fetch_sub(1, std::memory_order_acq_rel);

    return true;
}

bool ThreadPool::TrySteal(std::size_t thiefWorkerIndex, Task& task)
{
    if (m_WorkerCount == 0)
        return false;

    const auto startIndex = m_NextQueue.fetch_add(1, std::memory_order_relaxed) % m_WorkerCount;

    for (std::size_t offset{}; offset < m_WorkerCount; ++offset)
    {
        const auto queueIndex = (startIndex + offset) % m_WorkerCount;

        if (queueIndex == thiefWorkerIndex)
            continue;

        auto& queue = *m_Queues[queueIndex];

        std::unique_lock lock(queue.mutex, std::try_to_lock);

        if (!lock || queue.tasks.empty())
            continue;

        task = std::move(queue.tasks.front());

        queue.tasks.pop_front();

        m_QueuedTaskCount.fetch_sub(1, std::memory_order_acq_rel);

        return true;
    }

    return false;
}

void ThreadPool::WaitForCompletion(CompletionState& state)
{
    while (true)
    {
        const auto remaining = state.remaining.load(std::memory_order_acquire);

        if (remaining == 0)
            return;

        if (TryExecuteOne(InvalidWorkerIndex))
            continue;

        state.remaining.wait(remaining, std::memory_order_acquire);
    }
}

bool ThreadPool::InParallelRegion() const noexcept
{
    return s_CurrentPool == this || s_ParallelDepth != 0;
}
