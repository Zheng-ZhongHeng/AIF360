from aif360.metrics import RegressionDatasetMetric
from aif360.datasets import RegressionDataset
import numpy as np
import pandas as pd

df = pd.DataFrame([
    ['r', 55],
    ['b', 65],
    ['b', 85],
    ['b', 70],
    ['r', 60],
    ['r', 50],
    ['r', 40],
    ['b', 30],
    ['r', 20],
    ['b', 10],
], columns=['s', 'score'])
df['s'] = df['s'].astype(object)

dataset = RegressionDataset(df, dep_var_name='score', protected_attribute_names=['s'], privileged_classes=[['r']])
# sorted_dataset = RegressionDataset(df, dep_var_name='score', protected_attribute_names=['s'], privileged_classes=[['r']])


m = RegressionDatasetMetric(dataset=dataset,
                            privileged_groups=[{'s': 1}],
                            unprivileged_groups=[{'s': 0}])

def test_infeasible_index():
    actual = m.infeasible_index(target_prop={1: 0.5, 0: 0.5}, r=10)
    expected = (1, [3])
    assert actual == expected, f'Infeasible Index calculated wrong, got {actual}, expected {expected}'

def test_dcg():
    actual = m.discounted_cum_gain(normalized=False)
    expected = 2.6126967369231484
    assert abs(actual - expected) < 1e-6

def test_ndcg():
    actual = m.discounted_cum_gain(normalized=True, full_dataset=dataset)
    expected = 0.9205433036318259


# --- Pseudo R² tests (McFadden's) ---
# Build a small synthetic dataset with binary labels and probability scores.
# privileged group: s == 'r' (mapped to 1), unprivileged group: s == 'b' (mapped to 0)
# labels: binary (0 or 1); scores: predicted probabilities (0~1)
_df_r2 = pd.DataFrame({
    's':      ['r', 'r', 'r', 'r', 'b', 'b', 'b', 'b'],
    'label':  [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
})
_df_r2['s'] = _df_r2['s'].astype(object)
_dataset_r2 = RegressionDataset(
    _df_r2, dep_var_name='label',
    protected_attribute_names=['s'],
    privileged_classes=[['r']]
)
# privileged group: predictions close to true labels → higher McFadden R²
# unprivileged group: predictions less accurate → lower McFadden R²
_preds = np.array([[0.9], [0.8], [0.2], [0.1],   # privileged: good predictions
                   [0.6], [0.4], [0.6], [0.4]])   # unprivileged: poor predictions
_dataset_r2.scores = _preds

_m_r2 = RegressionDatasetMetric(
    dataset=_dataset_r2,
    privileged_groups=[{'s': 1}],
    unprivileged_groups=[{'s': 0}],
)


def test_pseudo_r2_overall():
    r2 = _m_r2.pseudo_r2()
    assert isinstance(r2, (float, np.floating)), f"Expected float, got {type(r2)}"
    assert r2 <= 1.0, f"McFadden R² should be <= 1, got {r2}"
    assert abs(r2 - 0.3667937806535049) < 1e-9, f"Unexpected overall McFadden R², got {r2}"


def test_pseudo_r2_privileged():
    r2 = _m_r2.pseudo_r2(privileged=True)
    assert isinstance(r2, (float, np.floating)), f"Expected float, got {type(r2)}"
    assert r2 > 0, f"Privileged McFadden R² should be > 0, got {r2}"
    assert abs(r2 - 0.7630344058337939) < 1e-9, f"Unexpected privileged McFadden R², got {r2}"


def test_pseudo_r2_unprivileged():
    r2 = _m_r2.pseudo_r2(privileged=False)
    assert isinstance(r2, (float, np.floating)), f"Expected float, got {type(r2)}"
    assert abs(r2 - (-0.029446844526784144)) < 1e-9, f"Unexpected unprivileged McFadden R², got {r2}"


def test_pseudo_r2_parity():
    parity = _m_r2.pseudo_r2_parity()
    expected = _m_r2.pseudo_r2(privileged=False) - _m_r2.pseudo_r2(privileged=True)
    assert abs(parity - expected) < 1e-9, (
        f"pseudo_r2_parity() = {parity}, "
        f"pseudo_r2(False) - pseudo_r2(True) = {expected}"
    )