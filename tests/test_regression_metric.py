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


# --- Pseudo R² tests ---
# Build a small synthetic dataset with known labels and scores (predictions).
# privileged group: s == 'r' (mapped to 1), unprivileged group: s == 'b' (mapped to 0)
_df_r2 = pd.DataFrame({
    's':      ['r', 'r', 'r', 'b', 'b', 'b'],
    'label':  [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
})
_df_r2['s'] = _df_r2['s'].astype(object)
_dataset_r2 = RegressionDataset(
    _df_r2, dep_var_name='label',
    protected_attribute_names=['s'],
    privileged_classes=[['r']]
)
# Overwrite scores with imperfect predictions (in the normalized [0,1] space)
# privileged group (s==1): predictions close to truth → high R²
# unprivileged group (s==0): predictions less accurate → lower R²
_preds = np.array([[0.0], [0.1], [0.2], [0.6], [0.9], [1.0]])  # shape (6,1)
_dataset_r2.scores = _preds

_m_r2 = RegressionDatasetMetric(
    dataset=_dataset_r2,
    privileged_groups=[{'s': 1}],
    unprivileged_groups=[{'s': 0}],
)


def test_pseudo_r2_overall():
    r2 = _m_r2.pseudo_r2()
    assert isinstance(r2, (float, np.floating)), f"Expected float, got {type(r2)}"
    assert r2 <= 1.0, f"R² should be <= 1, got {r2}"
    assert abs(r2 - 0.9142857142857143) < 1e-9, f"Unexpected overall R², got {r2}"


def test_pseudo_r2_privileged():
    r2 = _m_r2.pseudo_r2(privileged=True)
    assert isinstance(r2, (float, np.floating)), f"Expected float, got {type(r2)}"
    assert abs(r2 - 0.375) < 1e-9, f"Unexpected privileged R², got {r2}"


def test_pseudo_r2_unprivileged():
    r2 = _m_r2.pseudo_r2(privileged=False)
    assert isinstance(r2, (float, np.floating)), f"Expected float, got {type(r2)}"
    assert abs(r2 - 0.875) < 1e-9, f"Unexpected unprivileged R², got {r2}"


def test_pseudo_r2_parity():
    parity = _m_r2.pseudo_r2_parity()
    expected = _m_r2.pseudo_r2(privileged=False) - _m_r2.pseudo_r2(privileged=True)
    assert abs(parity - expected) < 1e-9, (
        f"pseudo_r2_parity() = {parity}, "
        f"pseudo_r2(False) - pseudo_r2(True) = {expected}"
    )