# High-Performance Bit-Packed AVL Tree (Numba Powered)

## Overview

This project implements a **high-performance AVL Tree** in Python with a design philosophy much closer to **systems programming** than traditional Python data structures.

Instead of relying on Python objects, recursion, or pointer-heavy node classes, this implementation:

- Stores the entire tree in a **contiguous NumPy array**
- Uses **manual memory management** (free list + index reuse)
- Packs each node into **exactly 128 bits (2 × uint64)**
- Is fully **Numba JIT-compiled** for near C-level performance
- Supports **parallel bulk search** using all CPU cores

The result is a deterministic, cache-friendly, allocation-free AVL tree suitable for **high-frequency operations**, **large datasets**, and **performance-critical workloads**.

This is not a “toy” AVL tree. It is deliberately engineered.

---

## Quick Usage Example

```python
import numpy as np
from AVLTreeArray import AVLTree, warmup

# Trigger JIT compilation (recommended before benchmarking)
warmup()

# Create an AVL tree with a fixed capacity
avl = AVLTree(size=1000)

# Insert values
for v in [30, 20, 10, 40, 50, 25]:
    avl.insert(v)

# Single search
idx = avl.search(25)
print("Index of 25:", idx)

# Bulk parallel search
queries = np.array([10, 25, 99], dtype=np.uint64)
results = avl.search_bulk(queries)
print("Bulk search results:", results)

# Remove a value
avl.remove(20)

# In-order traversal (sorted output, validation only)
print("Sorted values:", avl.inorder())

print(avl)
```

> ⚠️ **Note**  
> `inorder()` is intended for validation and debugging only.  
> Avoid using it in performance-critical benchmarks.

---

## Key Design Goals

- **Zero Python object allocation during operations**
- **Predictable memory layout**
- **Minimal per-node memory footprint**
- **Iterative algorithms only (no recursion)**
- **Explicit control over balancing and memory reuse**
- **Parallel read operations**
- **Production-grade correctness with validation utilities**

---

## Node Memory Layout (Bit Packing)

Each AVL node is packed into a fixed **128-bit layout**, split across two `uint64` values:

```
[value (62 bits) | left (30 bits) | right (30 bits) | height (6 bits)]
```

### Why Bit Packing?

- Improves **cache locality**
- Reduces memory overhead compared to Python objects
- Enables **branchless, inlined field access**
- Makes node copying and updates extremely cheap

### Field Constraints

| Field  | Bits | Max Value |
|------|------|-----------|
| value | 62 | 2⁶² - 1 |
| left  | 30 | ~1 billion indices |
| right | 30 | ~1 billion indices |
| height| 6  | Max height = 63 |

> The `left` pointer is intentionally split across both 64-bit integers to achieve optimal packing without padding.

---

## Data Structure Layout

- **Tree Storage**: `np.ndarray(shape=(N, 2), dtype=uint64)`
- **Index 0**: Reserved as `NULL`
- **Root**: Stored as an index into the array
- **Children**: Stored as integer indices (not pointers)

This mimics a **struct-of-arrays memory model** while keeping the interface clean.

---

## Manual Memory Management

This implementation does **not** rely on Python’s garbage collector.

Instead, it uses:

- `_free`: Next unused index
- `_free_list`: Stack of freed node indices
- `_free_list_top`: Stack pointer

### Benefits

- Deterministic behavior
- No memory fragmentation
- Stable indices over time
- Extremely fast insert/remove cycles

Nodes are recycled immediately after deletion.

---

## Insertion Logic

Insertion is fully **iterative** and follows these steps:

1. Traverse the tree like a standard BST
2. Record the traversal path in a preallocated array
3. Insert the node at the correct leaf
4. Walk back up the path
5. Update heights
6. Detect imbalance factors
7. Apply one of:
   - LL Rotation
   - LR Rotation
   - RR Rotation
   - RL Rotation
8. Reconnect rotated subtrees

All operations occur **in-place** on the array.

---

## Deletion Logic

Deletion is significantly more complex and carefully optimized.

### Supported Cases

- Leaf node
- Single-child node
- Two-child node (via in-order successor)

### Deletion Steps

1. Iterative search while recording the path
2. If two children:
   - Swap value with in-order successor
3. Physically remove target node
4. Push freed index into free list
5. Traverse upward:
   - Recompute heights
   - Apply rotations as needed

The tree remains balanced at all times.

---

## Rotations

Supported rotations:

- **Single Right Rotation (LL)**
- **Single Left Rotation (RR)**
- **Left-Right Rotation (LR)**
- **Right-Left Rotation (RL)**

Each rotation:

- Rewrites child indices
- Recomputes heights bottom-up
- Returns the new subtree root index

No recursion. No temporary objects.

---

## Search Capabilities

### Single Search

- Iterative BST traversal
- Fully inlined and branch-efficient
- Returns node index or `0`

### Bulk Parallel Search

```python
search_bulk(values: np.ndarray) -> np.ndarray
```

- Uses `numba.prange`
- Thread-safe traversal
- Scales across CPU cores
- Ideal for analytics and batch queries

---

## Performance Characteristics

| Feature | Complexity |
|------|-----------|
| Insert | O(log n) |
| Remove | O(log n) |
| Search | O(log n) |
| Bulk Search | O(log n) per query (parallel) |

### Practical Performance

- Near C-level speed after JIT warmup
- Extremely low memory overhead
- Stable performance under heavy mutation

---

## JIT Compilation Strategy

- All hot paths are decorated with `@njit(inline="always")`
- No Python objects inside loops
- Preallocated scratch buffers
- Warmup utility included

```python
warmup()
```

This ensures consistent benchmark results.

---

## When to Use This

This implementation is ideal if:

- You need **deterministic performance**
- You care about **memory layout**
- You handle **millions of operations**
- Python object overhead is unacceptable
- You want C-like control with Python syntax

It is **not** intended for casual scripting or small datasets.

---

## Final Notes

This project intentionally prioritizes:

- Explicit logic over abstraction
- Control over convenience
- Performance over readability

If you are comfortable with systems-level thinking, this AVL tree will feel familiar.

If not — that is by design.

---

## License

MIT License
