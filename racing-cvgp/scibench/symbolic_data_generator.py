import numpy as np
import json


class DataX(object):
    def __init__(self, vars_range_and_types):
        """
        """
        list_of_samplers = json.loads(vars_range_and_types)
        self.data_X_samplers = []
        for one_sample in list_of_samplers:
            if one_sample['name'] == 'Uniform':
                self.data_X_samplers.append(UniformSampling(one_sample['range'], one_sample['only_positive']))
            elif one_sample['name'] == 'LogUniform':
                self.data_X_samplers.append(LogUniformSampling(one_sample['range'], one_sample['only_positive']))
            elif one_sample['name'] == 'IntegerUniform':
                self.data_X_samplers.append(IntegerSampling(one_sample['range'], one_sample['only_positive']))

    def randn(self, sample_size):
        list_of_X = [one_sampler(sample_size) for one_sampler in self.data_X_samplers]
        return np.stack(list_of_X, axis=0).transpose()


class DefaultSampling(object):
    def __init__(self, name, range, only_positive=False):
        self.name = name
        self.range = range
        self.only_positive = only_positive


class LogUniformSampling(DefaultSampling):
    def __init__(self, ranges, only_positive=False):
        super().__init__('LogUniform', ranges, only_positive)

    def __call__(self, sample_size):
        if self.only_positive:
            # x ~ U(0.1, 10.0)
            log10_min = np.log10(self.range[0])
            log10_max = np.log10(self.range[1])
            return 10.0 ** np.random.uniform(log10_min, log10_max, size=sample_size)
        else:
            # x ~ either U(0.0, 1.0) or U(-1.0, 0.) with 50% chance
            num_positives = sum(np.random.uniform(0.0, 1.0, size=sample_size) > 0.5)
            num_negatives = sample_size - num_positives
            log10_min = np.log10(self.range[0])
            log10_max = np.log10(self.range[1])
            pos_samples = 10.0 ** np.random.uniform(log10_min, log10_max, size=num_positives)
            neg_samples = -10.0 ** np.random.uniform(log10_min, log10_max, size=num_negatives)
            all_samples = np.concatenate([pos_samples, neg_samples])
            np.random.shuffle(all_samples)
            return all_samples


class UniformSampling(DefaultSampling):
    def __init__(self, ranges, only_positive=False):
        super().__init__('Uniform', ranges, only_positive)

    def __call__(self, sample_size):
        if self.only_positive:
            # x ~ U(0.0, 1.0)
            return np.random.uniform(self.range[0], self.range[1], size=sample_size)
        else:
            num_positives = sum(np.random.uniform(0.0, 1.0, size=sample_size) > 0.5)
            num_negatives = sample_size - num_positives
            pos_samples = np.random.uniform(self.range[0], self.range[1], size=num_positives)
            neg_samples = -np.random.uniform(self.range[0], self.range[1], size=num_negatives)
            all_samples = np.concatenate([pos_samples, neg_samples])
            np.random.shuffle(all_samples)
            return all_samples


class IntegerSampling(DefaultSampling):
    def __init__(self, ranges, only_positive=False):
        ranges = [int(ranges[0]), int(ranges[1])]
        super().__init__('IntegerUniform', ranges, only_positive)

    def __call__(self, sample_size):
        if self.only_positive:
            # x ~ U(1, 100)
            return np.random.randint(self.range[0], self.range[1], size=sample_size)
        else:
            # x ~ either U(1, 100) or U(-100, -1) with 50% chance
            num_positives = sum(np.random.uniform(0.0, 1.0, size=sample_size) > 0.5)
            num_negatives = sample_size - num_positives
            pos_samples = np.random.randint(self.range[0], self.range[1], size=num_positives)
            neg_samples = -np.random.randint(self.range[0], self.range[1], size=num_negatives)
            all_samples = np.concatenate([pos_samples, neg_samples])
            np.random.shuffle(all_samples)
            return all_samples

#
