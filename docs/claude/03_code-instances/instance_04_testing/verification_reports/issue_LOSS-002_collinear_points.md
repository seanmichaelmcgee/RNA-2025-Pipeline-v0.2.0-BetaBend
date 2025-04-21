# Component Issue Report

## Issue Information
- **Issue ID**: LOSS-002
- **Component**: stable_kabsch_align (Collinear Points)
- **Provider Instance**: 03_integration
- **Verification Date**: 2025-04-21
- **Severity**: Medium
- **Issue Type**: Functional
- **Status**: Open

## Issue Description
The `stable_kabsch_align` function fails to correctly handle collinear point sets, which are a degenerate case where all points lie along a single line. This causes the SVD (Singular Value Decomposition) calculation to produce a rank-deficient covariance matrix, resulting in incorrect alignments for these special cases.

## Expected Behavior
The Kabsch algorithm should correctly align collinear point sets by detecting the special case and implementing appropriate handling for the rank-deficient covariance matrix. After alignment, the collinear points should match their reference positions as closely as possible.

## Actual Behavior
When aligning collinear point sets, the algorithm produces incorrect alignments that do not match the reference positions. The test case `test_stable_kabsch_degenerate_collinear` fails because the SVD computation doesn't properly handle the rank deficiency in the covariance matrix that occurs when points are collinear.

## Reproduction Steps
1. Create a collinear point set along the x-axis
2. Create a second point set by rotating the first set (e.g., 90 degrees around z-axis)
3. Apply the `stable_kabsch_align` function to align the rotated set to the original
4. Verify that the aligned coordinates match the original coordinates

## Test Code
```python
def test_stable_kabsch_degenerate_collinear(self, device):
    """
    Test Kabsch alignment for collinear points.
    
    This test verifies that the Kabsch algorithm correctly handles
    the degenerate case of collinear points, where the covariance
    matrix will be rank deficient.
    
    The test creates points aligned along the x-axis, rotates them 90
    degrees around the z-axis, and then tries to align them back.
    """
    points_collinear = torch.tensor([[0.,0.,0.], [1.,0.,0.], [2.,0.,0.], [3.,0.,0.]], dtype=torch.float32, device=device)
    R = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], device=device)
    points_collinear_rotated = torch.matmul(points_collinear, R)
    aligned = stable_kabsch_align(points_collinear_rotated, points_collinear)
    assert torch.allclose(aligned, points_collinear, atol=1e-6)
```

## Evidence
The test is marked with `@pytest.mark.xfail` in the test file with the comment:
```python
@pytest.mark.xfail(reason="Known issue: Colinear points handling still needs improvement")
```

When running the test without the xfail marker, it fails with an assertion error because the aligned points do not match the original collinear points within the specified tolerance.

## Impact Assessment
- **Functional Impact**: Reduced accuracy in structure alignment for RNA sections that are nearly or perfectly collinear, which could affect FAPE loss calculation.
- **Integration Impact**: May propagate to downstream components that rely on accurate alignment, particularly affecting models of extended helical structures.
- **Performance Impact**: None; this is a correctness issue, not a performance issue.
- **Timeline Impact**: Low; the issue does not block functionality but should be addressed before final release.

## Root Cause Analysis
The issue occurs in the SVD step of the Kabsch algorithm when dealing with collinear point sets. The covariance matrix becomes rank-deficient (not full rank) in these cases, and the current implementation doesn't appropriately handle this special case. 

When points are collinear, one of the singular values becomes very close to zero, and numerical precision issues can cause incorrect rotation matrix calculations. The algorithm needs to explicitly detect this case and implement special handling for the rank-deficient rotation matrix.

## Recommended Resolution
1. Add explicit detection of collinear point sets by checking if any singular value is close to zero
2. Implement special handling for the rank-deficient case:
   ```python
   # After computing SVD of covariance matrix
   # Check for rank deficiency
   if torch.any(s < epsilon):
       # Handle the collinear case by constructing rotation matrix with special care
       # One approach: use QR decomposition as a fallback for these degenerate cases
       q, r = torch.linalg.qr(covariance.T)
       rotation = q @ torch.eye(3, device=q.device)
       
       # Ensure det(rotation) = 1 for a proper rotation
       if torch.det(rotation) < 0:
           rotation[:, 2] = -rotation[:, 2]
   ```
3. Add specific tests for collinear cases along different axes to ensure the solution is general

## Workaround
Currently, there is no workaround other than avoiding perfectly collinear structures in test cases. In practical use, most RNA structures are not perfectly collinear, so this issue primarily affects test cases and rare structural motifs.

## Resolution Timeline
- **Priority**: Medium
- **Expected Resolution Date**: 2025-04-27
- **Re-verification Required**: Yes

## Resolution Notes
[To be filled when issue is resolved]

## Resolution Verification
[To be filled when resolution is verified]