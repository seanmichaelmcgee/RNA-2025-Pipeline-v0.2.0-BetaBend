# Component Issue Report

## Issue Information
- **Issue ID**: LOSS-001
- **Component**: stable_kabsch_align
- **Provider Instance**: 03_integration
- **Verification Date**: 2025-04-20
- **Severity**: Medium
- **Issue Type**: Functional
- **Status**: Open

## Issue Description
The `stable_kabsch_align` function has precision issues when handling rotations, particularly during singular value decomposition (SVD) and subsequent rotation matrix calculation. This causes the Kabsch alignment algorithm to produce slightly incorrect alignments for rotated point sets, leading to imprecise FAPE loss calculations.

## Expected Behavior
The Kabsch algorithm should correctly align point sets regardless of the initial rotation. For two point sets that differ only by rotation, the alignment should result in coordinates that match the reference set with high precision (within numerical tolerance).

## Actual Behavior
When aligning point sets that differ by rotation, the resulting aligned coordinates show discrepancies larger than the expected numerical tolerance. The test `test_kabsch_rotation` fails because the aligned coordinates do not match the reference coordinates within the specified tolerance of 1e-6.

## Reproduction Steps
1. Create a reference point set
2. Create a second point set by rotating the reference set (e.g., 90 degrees around Z-axis)
3. Apply the `stable_kabsch_align` function to align the rotated set to the reference
4. Verify that the aligned coordinates match the reference coordinates

## Test Code
```python
def test_kabsch_rotation(self, points_a, points_b_rotated):
    """
    Test Kabsch alignment for rotated points.
    
    This test verifies that the Kabsch algorithm correctly aligns
    points that have been rotated around a center.
    """
    aligned_b = stable_kabsch_align(points_b_rotated, points_a)
    assert torch.allclose(aligned_b, points_a, atol=1e-6)
```

## Evidence
The test is marked with `@pytest.mark.xfail` in the test file with the comment:
```python
@pytest.mark.xfail(reason="Known issue: Kabsch rotation handling still has precision issues")
```

The test fails because the maximum difference between aligned and reference coordinates exceeds the tolerance of 1e-6.

## Impact Assessment
- **Functional Impact**: Reduced accuracy in FAPE loss calculation for structures that primarily differ by rotation, potentially leading to suboptimal model training.
- **Integration Impact**: The imprecision may propagate to downstream components that rely on accurate alignment, affecting overall model performance.
- **Performance Impact**: None; this is an accuracy issue, not a performance issue.
- **Timeline Impact**: Low; the issue does not block functionality but should be addressed before final release.

## Root Cause Analysis
The issue likely occurs in the SVD step of the Kabsch algorithm, where numerical precision errors in the computation of the rotation matrix can accumulate. The current implementation may not adequately handle the special case where the determinant of the rotation matrix is negative (a reflection rather than pure rotation).

Specifically, the issue appears in the handling of the sign of the determinant and subsequent correction for reflections, which is a known edge case in Kabsch implementations.

## Recommended Resolution
1. Revise the SVD computation in `stable_kabsch_align` to ensure higher precision
2. Add explicit handling for the case where the determinant of the rotation matrix is negative
3. Implement the correction for reflections as described in the Kabsch algorithm literature:
   ```python
   # Check if rotation matrix is a reflection
   if torch.det(rotation) < 0:
       # Correct for reflection by flipping the sign of the last column of V
       V[:, -1] = -V[:, -1]
       rotation = torch.mm(V, U.transpose(0, 1))
   ```
4. Consider updating the tolerance level in tests if necessary, while ensuring it remains strict enough to catch actual issues

## Workaround
Currently, the issue is marked as xfail in tests. In practical use, the alignment is still functional but with reduced precision. For critical applications requiring high-precision alignment, consider post-processing the aligned coordinates or implementing an alternative alignment algorithm.

## Resolution Timeline
- **Priority**: Medium
- **Expected Resolution Date**: 2025-04-27
- **Re-verification Required**: Yes

## Resolution Notes
[To be filled when issue is resolved]

## Resolution Verification
[To be filled when resolution is verified]