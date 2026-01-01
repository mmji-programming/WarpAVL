import numpy as np
from numba import njit, prange, uint64, int64
from numba.experimental import jitclass
from typing import Tuple



# Packed 128-bit layout:
#     LAYOUT[128]: [value[62] | left[30] | right[30] | height[6]]
#     Limitations:
#         0 <= value  <= (1 << 62) - 1
#         0 <= left   <= (1 << 30) - 1
#         0 <= right  <= (1 << 30) - 1
#         0 <= height <= (1 << 6)  - 1



# LAYOUT[128]: [value[62] | left[30] | right[30] | height[6]]
LEFT_LOW_MASK   = np.uint64(0xFFFFFFF)          # (1 << 28) - 1
LEFT_HIGH_MASK  = np.uint64(0x3)                # (1 <<  2) - 1
HEIGHT_MASK     = np.uint64(0x3F)               # (1 <<  6) - 1
RIGHT_MASK      = np.uint64(0x3FFFFFFF)         # (1 << 30) - 1
LEFT_MASK       = np.uint64(0x3FFFFFFF)         # (1 << 30) - 1
VALUE_MASK      = np.uint64(0x3FFFFFFFFFFFFFFF) # (1 << 62) - 1
LEFT_HIGH_SHIFT = np.uint64(0x1C) # 28
RIGHT_SHIFT     = np.uint64(0x6)  # 6
LEFT_SHIFT      = np.uint64(0x24) # 36
VALUE_SHIFT     = np.uint64(0x2)  # 2



# ---------- JIT-Compiled Bitwise Accessors / Updaters for Packed Fields ----------
@njit(inline="always")
def pack(
    value:  np.uint64,
    left:   np.uint64,
    right:  np.uint64,
    height: np.uint64
    
) -> Tuple[np.uint64, np.uint64]:
    
    """
    Pack four unsigned 64-bit integers into a tuple of two 64-bit integers (high, low) 
    according to a 128-bit layout: [value[62] | left[30] | right[30] | height[6]].

    The `left` field is split across both integers:
    - the lower 2 bits go into the high bits of `low`
    - the remaining 28 bits go into the lower bits of `high`.

    All fields are treated as unsigned 64-bit integers (`np.uint64`).

    :param value: The main value to pack (up to 62 bits)
    :type value: np.uint64
    :param left: The left field to pack (up to 30 bits)
    :type left: np.uint64
    :param right: The right field to pack (up to 30 bits)
    :type right: np.uint64
    :param height: The height field to pack (up to 6 bits)
    :type height: np.uint64
    :return: A tuple of two 64-bit unsigned integers (high, low) representing the packed 128-bit value
    :rtype: Tuple[np.uint64, np.uint64]
    """
    
    low  = ((left & LEFT_LOW_MASK) << LEFT_SHIFT) | ((right & RIGHT_MASK) << RIGHT_SHIFT) | (height & HEIGHT_MASK)
    high = ((value & VALUE_MASK) << VALUE_SHIFT) | ((left & LEFT_MASK) >> LEFT_HIGH_SHIFT)

    return np.uint64(high), np.uint64(low)

@njit(inline="always")
def unpack(
    high: np.uint64, 
    low:  np.uint64
    
) -> Tuple[np.uint64, np.uint64, np.uint64, np.uint64]:

    """
    Unpack a 128-bit value represented as a tuple of two 64-bit unsigned integers
    (high, low) into its original fields according to the layout: 
    [value[62] | left[30] | right[30] | height[6]].

    The `left` field is reconstructed from both integers:
    - 2 upper bits from `high`
    - 28 lower bits from `low`

    All fields are treated as unsigned 64-bit integers (`np.uint64`).

    NOTE:
    This function is intended **only for control, testing, or debugging purposes**.
    It is not meant to be used directly in performance-critical production code.
    For main execution, use packed `high` and `low` values without unpacking.

    :param high: The high 64-bit part of the packed value
    :type high: np.uint64
    :param low: The low 64-bit part of the packed value
    :type low: np.uint64
    :return: A tuple of four unsigned integers (value, left, right, height)
    :rtype: Tuple[np.uint64, np.uint64, np.uint64, np.uint64]
    """
    
    height = (low & HEIGHT_MASK)
    right  = (low >> RIGHT_SHIFT) & RIGHT_MASK
    left   = ((high & LEFT_HIGH_MASK) << LEFT_HIGH_SHIFT) | ((low >> LEFT_SHIFT) & LEFT_LOW_MASK)
    value  = (high >> VALUE_SHIFT) & VALUE_MASK

    return value, left, right, height

@njit(inline="always")
def _get_height(
    _:   np.uint64,
    low: np.uint64
    
) -> np.uint64:
    
    """
    Extract the 'height' field (6 bits) from the low 64-bit integer.
    """
    
    return np.uint64(low & HEIGHT_MASK)

@njit(inline="always")
def _get_right(
    _:   np.uint64,
    low: np.uint64
    
) -> np.uint64:
    
    """
    Extract the 'right' field (30 bits) from the low 64-bit integer.
    """
    
    return np.uint64((low >> RIGHT_SHIFT) & RIGHT_MASK)

@njit(inline="always")
def _get_left(
    high: np.uint64,
    low:  np.uint64
    
) -> np.uint64:
    
    """
    Reconstruct the 'left' field (30 bits) from both high and low 64-bit integers.
    """
    
    return np.uint64(((high & LEFT_HIGH_MASK) << LEFT_HIGH_SHIFT) | ((low >> LEFT_SHIFT) & LEFT_LOW_MASK))

@njit(inline="always")
def _get_value(
    high: np.uint64,
    _:    np.uint64
    
) -> np.uint64:
    
    """
    Extract the 'value' field (62 bits) from the high 64-bit integer.
    """
    
    return np.uint64((high >> VALUE_SHIFT) & VALUE_MASK)

@njit(inline="always")
def _update_height(
    _:          np.uint64,
    low:        np.uint64,
    new_height: np.uint64

) -> Tuple[np.uint64, np.uint64]:

    """
    Update the 'height' field (6 bits) in the low 64-bit integer.

    The high 64-bit integer is left unchanged and returned as-is.
    Only the height bits inside `low` are replaced with `new_height`.
    """
    
    low        = np.uint64(low)
    new_height = np.uint64(new_height)
    
    low = (low & ~HEIGHT_MASK) | (new_height & HEIGHT_MASK)
    
    return _, low

@njit(inline="always")
def _update_right(
    _:         np.uint64,
    low:       np.uint64,
    new_right: np.uint64
    
) -> Tuple[np.uint64, np.uint64]:
    
    """
    Update the 'right' field (30 bits) in the low 64-bit integer.

    The high 64-bit integer is left unchanged and returned as-is.
    Only the right bits inside `low` are replaced with `new_right`.
    """
    
    new_right = np.uint64(new_right) 
    low       = np.uint64(low)
    
    low = (low & ~(RIGHT_MASK << RIGHT_SHIFT)) | ((new_right & RIGHT_MASK) << RIGHT_SHIFT)
    
    return _, low

@njit(inline="always")
def _update_left(
    high:     np.uint64,
    low:      np.uint64,
    new_left: np.uint64
    
) -> Tuple[np.uint64, np.uint64]:

    """
    Update the 'left' field (30 bits) across both high and low 64-bit integers.

    The 5 lower bits of 'left' are stored in the high bits of `low`,
    and the remaining 2 higher bits are stored in the low bits of `high`.
    Only the left bits are modified; all other bits remain unchanged.

    :param high: The high 64-bit integer containing part of 'left' and 'value'
    :type high: np.uint64
    :param low: The low 64-bit integer containing part of 'left', 'right', and 'height'
    :type low: np.uint64
    :param new_left: The new 30-bit value to set for 'left'
    :type new_left: np.uint64
    :return: A tuple of updated (high, low) 64-bit integers
    :rtype: Tuple[np.uint64, np.uint64]
    """
    
    high     = np.uint64(high)
    low      = np.uint64(low)
    new_left = np.uint64(new_left)
    
    new_left_low  = (new_left & LEFT_MASK) & LEFT_LOW_MASK
    new_left_high = (new_left & LEFT_MASK) >> LEFT_HIGH_SHIFT
    
    low  = (low & ~(LEFT_LOW_MASK << LEFT_SHIFT)) | (new_left_low << LEFT_SHIFT)
    high = (high & (~LEFT_HIGH_MASK)) | (new_left_high & LEFT_HIGH_MASK)
    
    return high, low

@njit(inline="always")
def _update_value(
    high:      np.uint64,
    _:         np.uint64,
    new_value: np.uint64

) -> Tuple[np.uint64, np.uint64]:

    """
    Update the 'value' field (62 bits) in the high 64-bit integer.

    Only the top 62 bits representing 'value' are modified; the lower 2 bits (left_high) remain unchanged.
    The low 64-bit integer is returned as-is.

    :param high: The high 64-bit integer containing 'value' and part of 'left'
    :type high: np.uint64
    :param _: Placeholder for low (unused)
    :type _: np.uint64
    :param new_value: The new 62-bit value to set
    :type new_value: np.uint64
    :return: A tuple of updated (high, low) 64-bit integers
    :rtype: Tuple[np.uint64, np.uint64]
    """
    
    high      = np.uint64(high)
    new_value = np.uint64(new_value)
    
    high = (high & LEFT_HIGH_MASK) | ((new_value & VALUE_MASK) << VALUE_SHIFT)

    return high, _

@njit(inline="always")
def set_node(
    tree:  np.ndarray,
    index,
    node:  Tuple[np.uint64, np.uint64]
    
) -> None:

    """
    Assign a packed node tuple (high, low) to the given row in the tree array.
    Compatible with Numba nopython mode.
    """
    
    tree[np.int64(index), 0] = node[0]
    tree[np.int64(index), 1] = node[1]

@njit(inline="always")
def get_node(
    tree:  np.ndarray, 
    index: np.uint64

) -> Tuple[np.uint64, np.uint64]:

    """
    Get a node from tree by index.
    """
    
    i = np.uint64(index)
    return tree[i, 0], tree[i, 1]



# ---------- JIT-Compiled AVLTree Core Operations ----------
@njit(inline="always")
def get_successor(
    tree:       np.ndarray,
    index:      np.uint64,
    path:       np.ndarray,
    path_index: np.int64
) -> Tuple[np.uint64, np.int64]:
    
    """
    Locates the in-order successor of a node and updates the traversal path.

    In an AVL tree, the in-order successor is the smallest node in the right 
    subtree. This function moves to the right child and then follows the left 
    pointers to the leafmost node, while simultaneously recording these nodes 
    in the 'path' array to ensure proper rebalancing after the swap.

    Args:
        tree (np.ndarray): 2D array [N, 2] containing the packed AVL nodes.
        index (np.uint64): The index of the node whose successor is needed.
        path (np.ndarray): Array to store the traversal path for rebalancing.
        path_index (np.int64): The current write position in the path array.

    Returns:
        Tuple[np.uint64, np.int64]:
            - successor_index: The index of the in-order successor.
            - updated_path_index: The new path_index after adding the successor's lineage.
    """
    
    h, l = get_node(tree, index)
    curr = _get_right(h, l)
    
    while curr != 0:
        path[path_index] = curr
        path_index += 1
        h, l = get_node(tree, curr)
        curr = _get_left(h, l)
        
    return np.uint64(path[path_index - 1]), path_index

@njit(inline="always")
def right_rotation( # SRR: Single Right Rotation
    tree:  np.ndarray,
    index: np.uint64

) -> np.uint64:

    """
    Perform a single right rotation (SRR) on an AVL tree stored as a NumPy array.

    This operation rotates the node at the given `index` with its left child, 
    updating the parent-child relationships and recalculating node heights.

    The AVL tree nodes are packed in 64-bit integers, where left, right, value, 
    and height are encoded.

    :param tree: NumPy array representing the AVL tree, each element is a packed node
    :type tree: np.ndarray
    :param index: Index of the node to rotate
    :type index: int
    :return: Index of the new root of the rotated subtree
    :rtype: np.uint64
    """
      
    
    # Get nodes
    high_target_node, low_target_node = get_node(tree, index)
    left_index                        = _get_left(high_target_node, low_target_node)
    high_left_node, low_left_node     = get_node(tree, left_index)
    
    
    # Rotate 
    set_node(tree, index, pack(
        _get_value(high_target_node, low_target_node),
        _get_right(high_left_node, low_left_node),
        _get_right(high_target_node, low_target_node),
        np.uint64(0) # update later
    ))
    
    set_node(tree, left_index, pack(
        _get_value(high_left_node, low_left_node),
        _get_left(high_left_node, low_left_node),
        index,
        np.uint64(0) # update later
    ))
    
    # Update heightes
    # Update old root (index)
    h_idx, l_idx     = get_node(tree, index)
    h_idx_l, l_idx_l = get_node(tree, _get_left(h_idx, l_idx))
    h_idx_r, l_idx_r = get_node(tree, _get_right(h_idx, l_idx))
    
    set_node(tree, index, _update_height(
        h_idx, l_idx,
        new_height=max(
            _get_height(h_idx_l, l_idx_l),
            _get_height(h_idx_r, l_idx_r)
        ) + 1
    ))
    
    # Update new root (left_index)
    h_l_idx, l_l_idx     = get_node(tree, left_index)
    h_l_idx_l, l_l_idx_l = get_node(tree, _get_left(h_l_idx, l_l_idx))
    h_l_idx_r, l_l_idx_r = get_node(tree, _get_right(h_l_idx, l_l_idx))
    
    set_node(tree, left_index, _update_height(
        h_l_idx, l_l_idx,
        new_height=max(
            _get_height(h_l_idx_l, l_l_idx_l),
            _get_height(h_l_idx_r, l_l_idx_r)
            
        ) + 1
    ))
    
    return np.uint64(left_index) # new root

@njit(inline="always")
def left_rotation( # SLR: Single Left Rotation
    tree: np.ndarray,
    index: np.uint64
    
) -> np.uint64:

    """
    Perform a single left rotation (SLR) on an AVL tree stored as an array.

    This rotation is applied when a node becomes right-heavy.
    The right child of the target node becomes the new root of the subtree,
    and the target node becomes the left child of that node.

    The function:
    - Rewires the subtree pointers using packed node representation
    - Recomputes heights bottom-up according to AVL rules
    - Modifies the tree in-place

    :param tree: Array-based AVL tree containing packed nodes
    :type tree: np.ndarray
    :param index: Index of the subtree root to rotate
    :type index: np.uint64
    :return: Index of the new root after rotation
    :rtype: np.uint64
    """
    
    
    # Gets nodes
    high_target_node, low_target_node = get_node(tree, index)
    right_index                       = _get_right(high_target_node, low_target_node)
    high_right_node, low_right_node   = get_node(tree, right_index)

    
    # Rotate
    set_node(tree, index, pack(
        _get_value(high_target_node, low_target_node),
        _get_left(high_target_node, low_target_node),
        _get_left(high_right_node, low_right_node ),
        np.uint64(0) # update later
    ))
    
    set_node(tree, right_index, pack(
        _get_value(high_right_node, low_right_node ),
        index,
        _get_right(high_right_node, low_right_node ),
        np.uint64(0) # update later
    ))
    
    # Update heights
    # height(index) = max(height(index->left), height(index->right)) + 1
    h_idx, l_idx     = get_node(tree, index)
    h_idx_l, l_idx_l = get_node(tree, _get_left(h_idx, l_idx))
    h_idx_r, l_idx_r = get_node(tree, _get_right(h_idx, l_idx))
    
    set_node(tree, index, _update_height(
        h_idx, l_idx,
        new_height=max(
            _get_height(h_idx_l, l_idx_l),
            _get_height(h_idx_r, l_idx_r)
            
        ) + 1
    ))
    
    # height(right_index) = max(height(right_index->left), height(right_index->right)) + 1
    h_r_idx, l_r_idx     = get_node(tree, right_index)
    h_r_idx_l, l_r_idx_l = get_node(tree, _get_left(h_r_idx, l_r_idx))
    h_r_idx_r, l_r_idx_r = get_node(tree, _get_right(h_r_idx, l_r_idx))
    
    set_node(tree, right_index, _update_height(
        h_r_idx, l_r_idx,
        new_height=max(
            _get_height(h_r_idx_l, l_r_idx_l),
            _get_height(h_r_idx_r, l_r_idx_r)    
        
        ) + 1
    ))
    
    return np.uint64(right_index) # new root

@njit(inline="always", boundscheck=False)
def insert(
    tree:          np.ndarray,
    root:          np.uint64 ,
    free:          np.uint64 , # start from 1
    free_list:     np.ndarray,
    free_list_top: np.int64  ,
    path:          np.ndarray, # use in rebalancing
    value:         np.uint64
    
) -> Tuple[np.uint64, np.uint64, np.int64]:
    
    """
    Insert a new value into an array-based, bit-packed AVL tree with rebalancing.
    The tree is stored as a `np.ndarray` of 128-bit nodes, packed into two uint64 values
    (high, low) with the following layout:
        [value[62]|left[30]| right[30]|height[6]]

    Parameters
    ----------
    tree : np.ndarray
        The array holding all nodes of the AVL tree.
    root : np.uint64
        Index of the current root node (0 if tree is empty).
    free : np.uint64
        Next unused index in `tree` if free_list is empty.
    free_list : np.ndarray
        Stack of previously freed node indices for reuse.
    free_list_top : np.int64
        Top index of the free_list stack (0 if empty).
    path : np.ndarray
        Preallocated array to store the traversal path for bottom-up rebalancing.
    value : np.uint64
        The value to insert into the AVL tree.

    Returns
    -------
    Tuple[np.uint64, np.uint64, np.int64]
        Updated (root, free, free_list_top) after insertion.
    """
    
    # First node
    if root == 0:
        if free_list_top > 0:
            free_list_top -= 1
            free_index = np.uint64(free_list[free_list_top])
        else:
            free_index = free
            free += 1
        
        set_node(tree, free_index, pack(value, 0, 0, 1))
        root = free_index
        return np.uint64(root), np.uint64(free), np.int64(free_list_top)

    # Insert new_node
    current_node_index = root
    path_index         = 0
    while current_node_index != 0:
        path[path_index] = current_node_index
        path_index += 1
        
        curr_high, curr_low  = get_node(tree, current_node_index)
        current_value         = _get_value(curr_high, curr_low)
        
        if value == current_value:
            return np.uint64(root), np.uint64(free), np.int64(free_list_top)
        
        elif value > current_value: # Right
            right_node_index = _get_right(curr_high, curr_low)

            if right_node_index == 0:
                if free_list_top > 0:
                    free_list_top -= 1
                    free_index = np.uint64(free_list[free_list_top])
                else:
                    free_index = free
                    free += 1
                
                set_node(tree, free_index, pack(value, 0, 0, 1))
                set_node(tree, current_node_index, _update_right(curr_high, curr_low, new_right=free_index))
                
                path[path_index] = free_index
                path_index += 1
                break
            
            current_node_index = right_node_index
        
        else: # Left
            left_node_index = _get_left(curr_high, curr_low)

            if left_node_index == 0:
                if free_list_top > 0:
                    free_list_top -= 1
                    free_index = np.uint64(free_list[free_list_top])
                else:
                    free_index = free
                    free += 1
                
                set_node(tree, free_index, pack(value, 0, 0, 1))
                set_node(tree, current_node_index, _update_left(curr_high, curr_low, new_left=free_index))
                
                path[path_index] = free_index
                path_index += 1
                break
            
            current_node_index = left_node_index
        
    # Rebalancing
    for idx in range(path_index - 2, -1, -1):
        
        node_index          = np.uint64(path[idx])
        node_high, node_low = get_node(tree, node_index)
        left_index          = _get_left(node_high, node_low)
        right_index         = _get_right(node_high, node_low)
        
        h_l = np.uint64(0)
        h_r = np.uint64(0)
        
        if left_index != 0:
            hl_h, hl_l = get_node(tree, left_index)
            h_l        = _get_height(hl_h, hl_l)
            
        if right_index != 0:
            hr_h, hr_l = get_node(tree, right_index)
            h_r        = _get_height(hr_h, hr_l)
            
        new_h = max(h_l, h_r) + 1
        if _get_height(node_high, node_low) != new_h:
            node_high, node_low = _update_height(node_high, node_low, new_h)
            set_node(tree, node_index, (node_high, node_low))
        
        bf = np.int64(h_l) - np.int64(h_r)

        new_sub_root = np.uint64(node_index)

        if bf > 1: # L
            h_l_child, l_l_child = get_node(tree, left_index)
            left_value           = _get_value(h_l_child, l_l_child)
            
            if value < left_value: # LL
                new_sub_root = np.uint64(right_rotation(tree, node_index))
                
            else: # LR
                new_left            = left_rotation(tree, left_index)
                node_high, node_low = _update_left(node_high, node_low, new_left)
                set_node(tree, node_index, (node_high, node_low))
                new_sub_root = np.uint64(right_rotation(tree, node_index))

        elif bf < -1: # R
            h_r_child, l_r_child = get_node(tree, right_index)
            right_value          = _get_value(h_r_child, l_r_child)
            
            if value > right_value: # RR
                new_sub_root = np.uint64(left_rotation(tree, node_index))
                
            else: # RL
                new_right           = right_rotation(tree, right_index)
                node_high, node_low = _update_right(node_high, node_low, new_right)
                set_node(tree, node_index, (node_high, node_low))
                new_sub_root = np.uint64(left_rotation(tree, node_index))
        
        if new_sub_root != node_index:
            if idx > 0:
                parent_idx = np.uint64(path[idx-1])
                p_h, p_l = get_node(tree, parent_idx)
                if _get_left(p_h, p_l) == node_index:
                    set_node(tree, parent_idx, _update_left(p_h, p_l, new_sub_root))
                else:
                    set_node(tree, parent_idx, _update_right(p_h, p_l, new_sub_root))
            else:
                root = new_sub_root
            break
    
    return np.uint64(root), np.uint64(free), np.int64(free_list_top)

@njit(inline="always", boundscheck=False)
def remove(
    tree:           np.ndarray,
    root:           np.uint64,
    free_list:      np.ndarray,
    free_list_top:  np.int64,
    path:           np.ndarray,
    value:          np.uint64
    
) -> Tuple[np.uint8, np.uint64, np.uint64]:

    """
    Iterative AVL tree node deletion with manual memory management.

    Performs a non-recursive removal of a value from the array-based AVL tree. 
    The process involves:
    1. Path Discovery: Traverses to the target node while recording the traversal 
    history in 'path' to facilitate bottom-up rebalancing.
    2. Logical Deletion: Handles leaf, single-child, and two-child cases 
    (using in-order successor for the two-child case).
    3. Memory Recycling: Adds the physically removed node's index back to the 
    'free_list' and increments 'free_list_top'.
    4. Retracing & Rebalancing: Updates heights and performs necessary AVL 
    rotations (LL, LR, RR, RL) from the point of deletion up to the root.

    Args:
        tree (np.ndarray): 2D array [N, 2] storing packed node data.
        root (np.uint64): Index of the current tree root.
        free_list (np.ndarray): Stack of available indices for node recycling.
        free_list_top (np.int64): Current pointer to the top of the free_list.
        path (np.ndarray): Scratchpad array to store the ancestor indices.
        value (np.uint64): The target value to be removed.

    Returns:
        Tuple[np.uint8, np.uint64, np.uint64]: 
            - success_flag (1 if found/deleted, 0 otherwise).
            - new_root_index.
            - updated_free_list_top.
    """

    if root == 0:
        return np.uint8(0), np.uint64(root), np.int64(free_list_top)
    
    # Search
    path_index    = 0
    current_index = root

    while current_index != 0:
        path[path_index] = current_index
        path_index += 1
        
        h_curr, l_curr = get_node(tree, current_index)
        v_curr         = _get_value(h_curr, l_curr)
        
        if value == v_curr:
            break
        elif value < v_curr:
            current_index = _get_left(h_curr, l_curr)
        else:
            current_index = _get_right(h_curr, l_curr)
    else:
        return np.uint8(0), np.uint64(root), np.int64(free_list_top)

    target_idx         = current_index
    h_target, l_target = get_node(tree, target_idx)
    lt, rt             = _get_left(h_target, l_target), _get_right(h_target, l_target)

    actual_remove_idx = np.uint64(0)

    if lt != 0 and rt != 0:
        successor_idx, new_path_idx = get_successor(tree, target_idx, path, path_index)
        h_succ, l_succ              = get_node(tree, successor_idx)
        s_val                       = _get_value(h_succ, l_succ)
        
        # Reload target
        h_t, l_t = get_node(tree, target_idx)
        set_node(tree, target_idx, _update_value(h_t, l_t, s_val))
        
        actual_remove_idx = successor_idx
        path_index        = new_path_idx
    else:
        actual_remove_idx = target_idx

    h_rem, l_rem = get_node(tree, actual_remove_idx)
    rl           = _get_left(h_rem, l_rem)
    rr           = _get_right(h_rem, l_rem)
    replacement  = rl if rl != 0 else rr

    if path_index > 1:
        parent_idx = path[path_index - 2]
        hp, lp     = get_node(tree, parent_idx)
        if _get_left(hp, lp) == actual_remove_idx:
            set_node(tree, parent_idx, _update_left(hp, lp, replacement))
        else:
            set_node(tree, parent_idx, _update_right(hp, lp, replacement))
    else:
        root = replacement

    free_list[free_list_top] = actual_remove_idx
    free_list_top += 1
    
    # Rebalancing
    for i in range(path_index - 2, -1, -1):
        node_idx = path[i]
        h_node, l_node = get_node(tree, node_idx)
        cl_idx         = _get_left(h_node, l_node)
        cr_idx         = _get_right(h_node, l_node)
        
        h_l = np.uint64(0)
        h_r = np.uint64(0) 
        
        if cl_idx != 0:
            hlh, hll = get_node(tree, cl_idx)
            h_l      = _get_height(hlh, hll)
            
        if cr_idx != 0:
            hrh, hrl = get_node(tree, cr_idx)
            h_r      = _get_height(hrh, hrl)
            
        new_h = max(h_l, h_r) + 1
        
        if _get_height(h_node, l_node) != new_h:
            h_node, l_node = _update_height(h_node, l_node, new_h)
            set_node(tree, node_idx, (h_node, l_node))
        
        bf = np.int64(h_l) - np.int64(h_r)
        
        new_sub_root = node_idx

        if bf > 1: # L
            h_lc, l_lc = get_node(tree, cl_idx)
            ll_idx     = _get_left(h_lc, l_lc)
            lr_idx     = _get_right(h_lc, l_lc)
            h_ll       = np.uint64(0)
            h_lr       = np.uint64(0)
            if ll_idx != 0: 
                h_tmp_h, h_tmp_l = get_node(tree, ll_idx)
                h_ll             = _get_height(h_tmp_h, h_tmp_l)
            if lr_idx != 0:
                h_tmp_h, h_tmp_l = get_node(tree, lr_idx)
                h_lr             = _get_height(h_tmp_h, h_tmp_l)
            
            if np.int64(h_ll) - np.int64(h_lr) >= 0: # LL
                new_sub_root = right_rotation(tree, node_idx)
            else: # LR
                mid_root       = left_rotation(tree, cl_idx)
                h_node, l_node = _update_left(h_node, l_node, mid_root)
                set_node(tree, node_idx, (h_node, l_node))
                new_sub_root = right_rotation(tree, node_idx)

        elif bf < -1: # R
            h_rc, l_rc = get_node(tree, cr_idx)
            rl_idx     = _get_left(h_rc, l_rc)
            rr_idx     = _get_right(h_rc, l_rc)
            h_rl       = np.uint64(0)
            h_rr       = np.uint64(0)
            if rl_idx != 0:
                h_tmp_h, h_tmp_l = get_node(tree, rl_idx)
                h_rl             = _get_height(h_tmp_h, h_tmp_l)
                
            if rr_idx != 0:
                h_tmp_h, h_tmp_l = get_node(tree, rr_idx)
                h_rr             = _get_height(h_tmp_h, h_tmp_l)

            if np.int64(h_rl) - np.int64(h_rr) <= 0: # RR
                new_sub_root = left_rotation(tree, node_idx)
            else: # RL
                mid_root       = right_rotation(tree, cr_idx)
                h_node, l_node = _update_right(h_node, l_node, mid_root)
                set_node(tree, node_idx, (h_node, l_node))
                new_sub_root = left_rotation(tree, node_idx)
        
        if new_sub_root != node_idx:
            if i > 0:
                p_idx  = path[i-1]
                ph, pl = get_node(tree, p_idx)
                if _get_left(ph, pl) == node_idx:
                    set_node(tree, p_idx, _update_left(ph, pl, new_sub_root))
                else:
                    set_node(tree, p_idx, _update_right(ph, pl, new_sub_root))
            else:
                root = new_sub_root
            
    return np.uint8(1), np.uint64(root), np.int64(free_list_top)

@njit(inline="always")
def _search_single(
    tree:  np.ndarray,
    root:  np.uint64,
    value: np.uint64
    
) -> np.uint64:
    
    """
    Performs a fast iterative search for a single value in the array-based AVL tree.

    This is a low-level utility designed to be inlined into larger search loops. 
    It traverses the tree using binary search logic, leveraging packed bit data 
    for rapid node navigation without recursion overhead.

    Args:
        tree (np.ndarray): 2D array [N, 2] containing the packed AVL nodes.
        root (np.uint64): The index of the root node to start the search from.
        value (np.uint64): The specific value to locate within the tree.

    Returns:
        np.uint64: The index of the node containing the value if found; 
                otherwise, returns 0.
    """
    
    current_index = root
    while current_index != 0:
        high_curr, low_curr = get_node(tree, current_index)
        value_curr          = _get_value(high_curr, low_curr)

        if value == value_curr:
            return current_index
        
        elif value < value_curr:
            current_index = _get_left(high_curr, low_curr)
        
        else:
            current_index = _get_right(high_curr, low_curr)
    
    return np.uint64(0)

@njit(parallel=True)
def _search_bulk(
    tree:   np.ndarray,
    root:   np.uint64,
    values: np.ndarray

) -> np.ndarray[np.uint64]:
    
    """
    Executes high-performance parallel searches for multiple values across the AVL tree.

    This function utilizes Numba's 'prange' to distribute search queries across 
    all available CPU cores. It leverages the thread-safe nature of tree traversal 
    to perform concurrent '_search_single' calls, maximizing throughput for 
    large-scale lookups.

    Args:
        tree (np.ndarray): 2D array [N, 2] containing the packed AVL nodes.
        root (np.uint64): The index of the tree root (shared across all threads).
        values (np.ndarray): 1D array of target values (uint64) to search for.

    Returns:
        np.ndarray: A 1D array of uint64 indices corresponding to the position 
                    of each input value in the tree. Returns 0 for values not found.
    """
    
    size    = values.size
    results = np.zeros(size, dtype=np.uint64)
    for i in prange(size):
        results[i] = _search_single(
            tree, root, values[i]
        )
        
    return results



# --------- Utils ---------
@njit
def warmup(tree_size: int = 100):
    """
    Minimally triggers JIT compilation for core AVL operations.
    """
    
    avl         = AVLTree(tree_size)
    warmup_data = np.array([30, 20, 10, 40, 50, 25], dtype=np.uint64)
    
    for x in warmup_data:
        avl.insert(int(x))
    
    _ = avl.search(20)
    
    queries = np.array([10, 25, 99], dtype=np.uint64)
    _ = avl.search_bulk(queries)
    
    avl.remove(10)
    
    return True

@njit
def build_avl(
    data: np.ndarray
    
) -> 'AVLTree':
    
    """
    Builds and populates an AVLTree from a NumPy array at machine speed.
    
    Args:
        data (np.ndarray): 1D array of uint64 values to insert.
        
    Returns:
        AVLTree: A fully balanced tree containing all elements from data.
    """
    
    avl = AVLTree(data.size)

    for i in range(data.size):
        avl.insert(data[i])
    
    return avl

@njit
def fill_avl(
    avl:  'AVLTree',
    data: np.ndarray
    
) -> None:
    
    """
    Populates an existing AVLTree with multiple values in a high-performance JIT loop.
    
    Args:
        avl (AVLTree): An instance of the AVLTree class to be populated.
        data (np.ndarray): 1D array of uint64 values to be inserted into the tree.
    """
    
    for i in range(data.size):
        avl.insert(data[i])

@njit
def remove_avl(
    tree:   'AVLTree',
    values: np.ndarray
    
) -> None:
    """
    Perform batch removal of multiple values from the AVL tree.
    
    This function is JIT-compiled for high-performance sequential deletions.
    It iterates through the provided array and calls the tree's internal 
    remove method for each element, maintaining AVL balance at each step.

    Args:
        tree (AVLTree): The jitclass instance of the AVL Tree.
        values (np.ndarray): A 1D NumPy array containing the values to be removed.
        
    Note:
        If a value in the array does not exist in the tree, it will be 
        silently ignored (as per the tree.remove implementation).
    """
    
    for i in range(values.size):
        tree.remove(values[i])
    
@njit
def inorder_traversal( # LVR
    tree:         np.ndarray,
    root:         np.uint64,
    current_size: np.int64
    
) -> np.ndarray:
    """
    Extracts all tree nodes in ascending order.
    Recommended for integrity checks and debugging on small to medium datasets.
    """
    
    traverse = np.zeros(current_size, dtype=np.uint64)
    stack    = np.zeros(256, dtype=np.uint64)
    
    current_index = root
    stack_idx     = 0
    traverse_idx  = 0
    
    while traverse_idx < current_size:
        
        while current_index != 0:
            stack[stack_idx] = current_index
            stack_idx += 1
            
            h, l          = get_node(tree, current_index)
            current_index = _get_left(h, l)
        
        if stack_idx > 0:
            stack_idx -= 1
            current_index = stack[stack_idx]
            
            h, l                   = get_node(tree, current_index)
            traverse[traverse_idx] = _get_value(h, l)
            traverse_idx += 1
            
            current_index = _get_right(h, l)
            
        else:
            break
            
    return traverse
        
    
# --------- AVLTree API ---------
spec = [
    ("size"          , uint64),
    ("count"         , int64),
    ("tree"          , uint64[:, :]),
    ("root"          , uint64),
    ("_free"         , uint64),
    ("_free_list"    , uint64[:]),
    ("_free_list_top", int64),
    ("_path"         , uint64[:]),
    
]

@jitclass(spec)
class AVLTree:
    """
    High-performance, bit-packed AVL Tree implemented as a Numba jitclass.
    
    Uses a 128-bit node layout (2x uint64) stored in a NumPy array to achieve 
    minimal memory footprint and C-level traversal speeds. Features manual 
    memory management with a free-list and parallel bulk search capabilities.

    Attributes:
        size (uint64): Maximum allocated capacity of the tree.
        count (int64): Current number of active nodes in the tree.
        tree (uint64[:, :]): Underlying 2D array [size, 2] storing packed nodes.
        root (uint64): Index of the current root node (0 if empty).
    """
    
    def __init__(
        self, 
        size: int
        
    ) -> None:
        
        if not (0 <= size <= RIGHT_MASK):
            raise ValueError(
                f"The size value must be between 0 and {RIGHT_MASK}, not {size}"
            )
        
        
        self.size           = uint64(size + 1)
        self.count          = int64(0)
        self.tree           = np.zeros((self.size, 2), dtype=np.uint64)
        self.root           = uint64(0)
        self._free          = uint64(1)
        self._free_list     = np.zeros((self.size), dtype=np.uint64)
        self._free_list_top = int64(0)
        self._path          = np.zeros(256, dtype=np.uint64)
    
    @property
    def height(self) -> int:
        h_r, l_r = get_node(self.tree, self.root)
        height   = _get_height(h_r, l_r)

        return int(height)
    
    @property
    def root_info(self) -> Tuple[int, int, int, int]:
        h_r, l_r = get_node(self.tree, self.root)
        value, left, right, height = unpack(h_r, l_r)
        return value, left, right, height
    
    @property
    def _max(self) -> int:
        """
        Find the index of the node with the maximum value in the tree.
        
        Returns:
            int: The index of the rightmost node, or 0 if the tree is empty.
        """
        
        if self.root == 0:
            return 0
        
        current = self.root
        while 1:
            
            h, l  = get_node(self.tree, current)
            right = _get_right(h, l)

            if right == 0:
                return int(current)

            current = right
        
    @property
    def _min(self) -> int:
        """
        Find the index of the node with the minimum value in the tree.
        
        Returns:
            int: The index of the leftmost node, or 0 if the tree is empty.
        """
        
        if self.root == 0:
            return 0

        current = self.root
        while 1:
            
            h, l = get_node(self.tree, current)
            left = _get_left(h, l) 

            if left == 0:
                return int(current)
            
            current = left
    
    def get_node(
        self, 
        index: int
        
    ) -> Tuple[int, int, int, int]:
        
        """
        Unpack all metadata for a specific node index.
        
        Args:
            index (int): The index of the node in the tree array.
            
        Returns:
            Tuple[int, int, int, int]: (value, left_index, right_index, height).
        """
        
        if index == 0:
            return 0, 0, 0, 0
        
        h, l = get_node(self.tree, index)
        return unpack(h, l)

    def get_value(
        self,
        index: int
        
    ) -> int:
        
        """
        Retrieve the stored 62-bit value of a specific node.
        
        Args:
            index (int): The index of the node.
            
        Returns:
            int: The unsigned 64-bit integer value stored in the node.
        """
        
        if index == 0:
            return 0
        
        value, _, _, _ = self.get_node(index)
        return value

    def get_left(
        self, 
        index: int
        
    ) -> int:
        
        """
        Get the index of the left child for the given node.
        
        Args:
            index (int): The index of the parent node.
            
        Returns:
            int: The index of the left child, or 0 if no child exists.
        """
        
        if index == 0:
            return 0
        
        _, left, _, _ = self.get_node(index)
        return int(left)

    def get_right(
        self, 
        index: int
            
    ) -> int:
        """
        Get the index of the right child for the given node.
        
        Args:
            index (int): The index of the parent node.
            
        Returns:
            int: The index of the right child, or 0 if no child exists.
        """
        
        if index == 0:
            return 0
        
        _, _, right, _ = self.get_node(index)
        return int(right)

    def get_height(
        self,
        index: int
        
    ) -> int:
        
        """
        Get the current height of a specific node.
        
        Args:
            index (int): The index of the node.
            
        Returns:
            int: The height value used for AVL balancing.
        """
        
        if index == 0:
            return 0
        
        _, _, _, height = self.get_node(index)
        return int(height)
    
    def successor(
        self, 
        index: int
        
    ) -> int:
        """
        Find the in-order successor of a node within its own subtree.
        
        Args:
            index (int): The index of the node to start from.
            
        Returns:
            int: The index of the smallest node in the right subtree, 
                 or 0 if no right child exists.
        """
        
        h, l    = get_node(self.tree, index)
        current = _get_right(h, l)
        
        if current == 0:
            return 0
        
        while 1:
            
            h_l, l_l = get_node(self.tree, current)
            left     = _get_left(h_l, l_l)
            
            if left == 0:
                return int(current)
            
            current = left
    
    def predecessor(
        self, 
        index: int
        
    ) -> int:
        """
        Find the in-order predecessor of a node within its own subtree.
        
        Args:
            index (int): The index of the node to start from.
            
        Returns:
            int: The index of the largest node in the left subtree, 
                 or 0 if no left child exists.
        """

        h, l = get_node(self.tree, index)
        current = _get_left(h, l)

        if current == 0:
            return 0
        
        while 1:
            h_r, l_r = get_node(self.tree, current)
            right    = _get_right(h_r, l_r)

            if right == 0:
                return int(current)
            
            current = right
    
    def insert(
        self,
        value: int
        
    ) -> int:
        """Inserts a unique value with auto-rebalancing. Returns 1 if success, 0 if fail/duplicate."""
        
        if self.count >= int64(self.size - 1):
            return 0
                
        self.root, self._free, self._free_list_top = insert(
            self.tree,
            self.root,
            self._free,
            self._free_list,
            self._free_list_top,
            self._path,
            np.uint64(value)
        )
        
        self.count += 1
        return 1
    
    def remove(
        self,
        value: int
        
    ) -> int:
        """Deletes a value and stabilizes the tree. Returns 1 if found and removed, 0 otherwise."""
        
        if self.count == 0:
            return 0
        
        success, root, free_list_top = remove(
            self.tree,
            self.root,
            self._free_list,
            self._free_list_top,
            self._path,
            np.uint64(value)
        )
        
        if success:
            self.root           = root
            self._free_list_top = free_list_top
            self.count -= 1
            return 1

        return 0
    
    def search(
        self,
        value: int
        
    ) -> int:
        """Locates a value using iterative BST search. Returns the node index or 0 if not found."""
        
        return _search_single(
            self.tree,
            self.root,
            np.uint64(value)
        )
    
    def search_bulk(
        self,
        values: np.ndarray
        
    ) -> np.ndarray[np.uint64]:
        """Performs parallelized multi-value search using all available CPU cores."""
        
        return _search_bulk(
            self.tree,
            self.root,
            values
        )
    
    def update_value(
        self,
        old_value: int,
        new_value: int
        
    ) -> int:
        """Replaces a value by atomic removal and re-insertion to maintain AVL properties."""
        
        if self.remove(old_value):
            self.insert(new_value)
            return 1
        
        return 0

    def inorder(self) -> np.ndarray:
        """
        Generates a sorted array of all elements using In-order traversal.
        
        WARNING: This method is intended strictly for validation and integrity 
        checks on small to medium datasets. Avoid using it for large-scale 
        performance benchmarks as it involves full tree traversal and memory allocation.
        """
        return inorder_traversal(self.tree, self.root, self.count)
    
    def __len__(self) -> int:
        return int(self.count)
    
    def __str__(self) -> str:
        
        h = 0
        if self.root > 0:
            h, l = get_node(self.tree, self.root)
            h    = _get_height(h, l)
        
        return "AVLTree(size=" + str(self.count) + ", root=" + str(self.root) + ", height=" + str(int(h)) + ")"

