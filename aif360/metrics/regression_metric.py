import numpy as np
from aif360.metrics import DatasetMetric, utils
from aif360.datasets import RegressionDataset

_LOG_CLIP_EPS = 1e-10


class RegressionDatasetMetric(DatasetMetric):
    """Class for computing metrics based on a single
    :obj:`~aif360.datasets.RegressionDataset`.
    """

    def __init__(self, dataset, unprivileged_groups=None, privileged_groups=None):
        """
        Args:
            dataset (RegressionDataset): A RegressionDataset.
            privileged_groups (list(dict)): Privileged groups. Format is a list
                of `dicts` where the keys are `protected_attribute_names` and
                the values are values in `protected_attributes`. Each `dict`
                element describes a single group. See examples for more details.
            unprivileged_groups (list(dict)): Unprivileged groups in the same
                format as `privileged_groups`.

        Raises:
            TypeError: `dataset` must be a
                :obj:`~aif360.datasets.RegressionDataset` type.
        """
        if not isinstance(dataset, RegressionDataset):
            raise TypeError("'dataset' should be a RegressionDataset")

        # sets self.dataset, self.unprivileged_groups, self.privileged_groups
        super(RegressionDatasetMetric, self).__init__(dataset,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)
        
    def infeasible_index(self, target_prop: dict, r: int = None):
        """
        Infeasible Index metric, as described in [1]_.

        Args:
            target_prop (dict): desired proportion of groups.
            r (int): size of the candidate list over which the metric is calculated.
            Defaults to the size of the dataset.
        
        Returns:
            A tuple (int, set{int}): InfeasibleIndex and the positions at which the 
            feasibility condition is violated. 
        
    References:
        .. [1] Sahin Cem Geyik, Stuart Ambler, and Krishnaram Kenthapadi,
            "Fairness-Aware Ranking in Search & Recommendation Systems with Application to LinkedIn Talent Search,"
            KDD '19: Proceedings of the 25th ACM SIGKDD International Conference
            on Knowledge Discovery & Data Mining, July 2019, Pages 2221-2231.
        """
        pr_attr_values = np.ravel(
            self.dataset.unprivileged_protected_attributes + self.dataset.privileged_protected_attributes)
        if set(list(target_prop.keys())) != set(pr_attr_values):
            raise ValueError('Desired proportions must be specified for all values of the protected attributes!')
        
        ranking = np.column_stack((self.dataset.scores, self.dataset.protected_attributes))
        if r is None:
            r = np.ravel(self.dataset.scores).shape[0]
        ii = 0
        k_viol = set()
        for k in range(1, r):
            rk = ranking[:k]
            for ai in pr_attr_values:
                count_ai = rk[rk[:,1] == ai].shape[0]
                if count_ai < np.floor(target_prop[ai]*k):
                    ii+=1
                    k_viol.add(k-1)
        return ii, list(k_viol)
    
    def discounted_cum_gain(self, r: int = None, full_dataset: RegressionDataset=None, normalized=False):
        """
        Discounted Cumulative Gain metric.

        Args:
            r (int): position up to which to calculate the DCG. If not specified, is set to the size of the dataset.
            normalized (bool): return normalized DCG.
            
        Returns:
            The calculated DCG.
        """
        if r is None:
            r = np.ravel(self.dataset.scores).shape[0]
        if r < 0:
            raise ValueError(f'r must be >= 0, got {r}')
        if normalized == True and full_dataset is None:
            raise ValueError('`normalized` is set to True, but `full_dataset` is not specified')
        if not isinstance(full_dataset, RegressionDataset) and not (full_dataset is None):
            raise TypeError(f'`full_datset`: expected `RegressionDataset`, got {type(full_dataset)}')
        scores = np.ravel(self.dataset.scores)[:r]
        z = self._dcg(scores)
        if normalized:
            z /= self._dcg(np.sort(np.ravel(full_dataset.scores))[::-1][:r])
        return z
    
    def _dcg(self, scores):
        logs = np.log2(np.arange(2, len(scores)+2))
        z = np.sum(scores/logs)
        return z

    def pseudo_r2(self, privileged=None):
        """Compute McFadden's Pseudo R² for a group in a binary classification
        setting.

        .. math::

           R^2_{McFadden} = 1 - \\frac{\\ln L_{model}}{\\ln L_{null}}

        where

        .. math::

           \\ln L_{model} = \\sum_i \\left[ y_i \\ln(\\hat{p}_i) +
               (1 - y_i) \\ln(1 - \\hat{p}_i) \\right]

        and

        .. math::

           \\ln L_{null} = \\sum_i \\left[ y_i \\ln(\\bar{p}) +
               (1 - y_i) \\ln(1 - \\bar{p}) \\right]

        :math:`\\hat{p}_i` are the predicted probabilities from
        ``dataset.scores`` (values in ``[0, 1]``), :math:`y_i` are the binary
        true labels (``0`` or ``1``) from ``dataset.labels``, and
        :math:`\\bar{p}` is the base rate (mean of the labels) for the group.

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Returns:
            numpy.float64: McFadden's Pseudo R² value. Returns ``0.0`` if
            :math:`\\ln L_{null} = 0` (all labels are identical).
        """
        condition = self._to_condition(privileged)
        cond_vec = utils.compute_boolean_conditioning_vector(
            self.dataset.protected_attributes,
            self.dataset.protected_attribute_names,
            condition)

        y_true = np.ravel(self.dataset.labels)[cond_vec]
        y_pred = np.ravel(self.dataset.scores)[cond_vec]

        y_pred_clipped = np.clip(y_pred, _LOG_CLIP_EPS, 1 - _LOG_CLIP_EPS)
        ll_model = np.sum(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

        p_bar = np.mean(y_true)
        p_bar_clipped = np.clip(p_bar, _LOG_CLIP_EPS, 1 - _LOG_CLIP_EPS)
        ll_null = np.sum(y_true * np.log(p_bar_clipped) + (1 - y_true) * np.log(1 - p_bar_clipped))

        if ll_null == 0:
            return np.float64(0.0)

        return np.float64(1.0 - ll_model / ll_null)

    def pseudo_r2_parity(self):
        """Compute the difference in Pseudo R² between unprivileged and
        privileged groups.

        .. math::

           \\Delta R^2 = R^2_{\\text{unprivileged}} - R^2_{\\text{privileged}}

        A value of 0 indicates perfect fairness; a positive value indicates
        the model explains more variance for the unprivileged group; a negative
        value indicates the model explains more variance for the privileged
        group.

        Returns:
            numpy.float64: Difference in Pseudo R² (unprivileged − privileged).
        """
        return self.difference(self.pseudo_r2)
