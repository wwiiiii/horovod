# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import unittest
import warnings

import numpy as np

from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import DoubleType, LongType

import mock
import torch
import torch.nn as nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule

import horovod.spark.torch as hvd
from horovod.spark.common import constants, util
from horovod.spark.torch import remote
from horovod.spark.torch.estimator import EstimatorParams, _torch_param_serialize
from horovod.spark.torch.legacy import to_lightning_module

from spark_common import CallbackBackend, create_noisy_xor_data, create_xor_data, local_store, spark_session


class XOR(LightningModule):
    def __init__(self, input_dim=2, output_dim=1):
        super(XOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, 8)
        self.lin2 = nn.Linear(8, output_dim)

    def forward(self, features):
        x = features.float()
        x = self.lin1(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def training_step(self, batch, batch_nb):
        x, y = batch['features'], batch['y']
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.float())
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch['features'], batch['y']
        y_hat = self(x)
        return {'val_loss': F.binary_cross_entropy(y_hat, y.float())}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean() if len(outputs) > 0 else float('inf')
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


class LegacyXOR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LegacyXOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, 8)
        self.lin2 = nn.Linear(8, output_dim)

    def forward(self, features):
        x = features.float()
        x = self.lin1(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        return x


def create_xor_model(input_dim=2, output_dim=1):
    return XOR(input_dim, output_dim)


def create_legacy_xor_model(input_dim=2, output_dim=1):
    return LegacyXOR(input_dim, output_dim)


class SparkTorchTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SparkTorchTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def test_fit_model(self):
        model = create_xor_model()

        with spark_session('test_fit_model') as spark:
            df = create_noisy_xor_data(spark)

            with local_store() as store:
                torch_estimator = hvd.TorchEstimator(
                    num_proc=2,
                    store=store,
                    model=model,
                    input_shapes=[[2]],
                    feature_cols=['features'],
                    label_cols=['y'],
                    validation=0.2,
                    batch_size=4,
                    epochs=2,
                    verbose=2)

                torch_model = torch_estimator.fit(df)

                trained_model = torch_model.getModel()
                pred = trained_model(torch.ones([1, 2], dtype=torch.int32))
                assert len(pred) == 1
                assert pred.dtype == torch.float32

    def test_legacy_fit_model(self):
        model = create_legacy_xor_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = F.binary_cross_entropy

        with spark_session('test_fit_model') as spark:
            df = create_noisy_xor_data(spark)

            with local_store() as store:
                torch_estimator = hvd.TorchEstimator(
                    num_proc=2,
                    store=store,
                    model=model,
                    optimizer=optimizer,
                    loss=loss,
                    input_shapes=[[2]],
                    feature_cols=['features'],
                    label_cols=['y'],
                    batch_size=4,
                    epochs=2,
                    verbose=2,
                    sample_weight_col='weight')

                torch_model = torch_estimator.fit(df)

                trained_model = torch_model.getModel()
                pred = trained_model(torch.ones([1, 2], dtype=torch.int32))
                assert len(pred) == 1
                assert pred.dtype == torch.float32

    def test_restore_from_checkpoint(self):
        model = create_xor_model()

        with spark_session('test_restore_from_checkpoint') as spark:
            df = create_noisy_xor_data(spark)

            ctx = CallbackBackend()

            run_id = 'run01'
            with local_store() as store:
                torch_estimator = hvd.TorchEstimator(
                    backend=ctx,
                    store=store,
                    model=model,
                    input_shapes=[[2]],
                    feature_cols=['features'],
                    label_cols=['y'],
                    validation=0.2,
                    batch_size=4,
                    epochs=2,
                    verbose=2,
                    run_id=run_id)

                torch_estimator._read_checkpoint = mock.Mock(side_effect=torch_estimator._read_checkpoint)

                ckpt_path = store.get_checkpoint_path(run_id)
                assert not store.exists(ckpt_path)
                torch_estimator._read_checkpoint.assert_not_called()
                torch_estimator.fit(df)

                assert store.exists(ckpt_path)
                torch_estimator.fit(df)
                torch_estimator._read_checkpoint.assert_called()

    def test_legacy_restore_from_checkpoint(self):
        model = create_legacy_xor_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = nn.BCELoss()

        with spark_session('test_restore_from_checkpoint') as spark:
            df = create_noisy_xor_data(spark)

            ctx = CallbackBackend()

            run_id = 'run01'
            with local_store() as store:
                torch_estimator = hvd.TorchEstimator(
                    backend=ctx,
                    store=store,
                    model=model,
                    optimizer=optimizer,
                    loss=loss,
                    input_shapes=[[2]],
                    feature_cols=['features'],
                    label_cols=['y'],
                    validation=0.2,
                    batch_size=4,
                    epochs=2,
                    verbose=2,
                    run_id=run_id)

                torch_estimator._read_checkpoint = mock.Mock(side_effect=torch_estimator._read_checkpoint)

                ckpt_path = store.get_checkpoint_path(run_id)
                assert not store.exists(ckpt_path)
                torch_estimator._read_checkpoint.assert_not_called()
                torch_estimator.fit(df)

                assert store.exists(ckpt_path)
                torch_estimator.fit(df)
                torch_estimator._read_checkpoint.assert_called()

    def test_transform_multi_class(self):
        model = create_xor_model(output_dim=2)

        with spark_session('test_transform_multi_class') as spark:
            df = create_xor_data(spark)
            metadata = util._get_metadata(df)

            torch_model = hvd.TorchModel(history=None,
                                         model=model,
                                         input_shapes=[[2]],
                                         feature_columns=['features'],
                                         label_columns=['y'],
                                         _metadata=metadata)
            out_df = torch_model.transform(df)

            expected_types = {
                'x1': LongType,
                'x2': LongType,
                'features': VectorUDT,
                'weight': DoubleType,
                'y': DoubleType,
                'y__output': VectorUDT
            }

            for field in out_df.schema.fields:
                assert type(field.dataType) == expected_types[field.name]

    @mock.patch('horovod.torch.allgather')
    @mock.patch('horovod.torch.local_size')
    def test_calculate_shuffle_buffer_size_small_row_size(self, mock_local_size, mock_allgather):
        import horovod.torch as hvd
        hvd.init()

        hvd_size = 4
        local_size = 2
        mock_local_size.return_value = local_size
        mock_allgather.return_value = torch.tensor([local_size for _ in range(hvd_size)])

        avg_row_size = 100
        train_row_count_per_worker = 100

        calculate_shuffle_buffer_size = remote._calculate_shuffle_buffer_size_fn(
            train_row_count_per_worker, avg_row_size, None)
        shuffle_size = calculate_shuffle_buffer_size()
        assert shuffle_size == train_row_count_per_worker

    @mock.patch('horovod.torch.allgather')
    @mock.patch('horovod.torch.local_size')
    def test_calculate_shuffle_buffer_size(self, mock_local_size, mock_allgather):
        import horovod.torch as hvd
        hvd.init()

        # case with 2 workers, one with 5 ranks and second with 3 ranks
        mock_allgather.return_value = torch.tensor([5, 5, 5, 5, 5, 3, 3, 3])
        mock_local_size.return_value = 2

        avg_row_size = 100000
        train_row_count_per_worker = 1000000

        calculate_shuffle_buffer_size = remote._calculate_shuffle_buffer_size_fn(
            train_row_count_per_worker, avg_row_size, None)
        shuffle_size = calculate_shuffle_buffer_size()

        actual = int(shuffle_size)
        expected = int(constants.TOTAL_BUFFER_MEMORY_CAP_GIB * constants.BYTES_PER_GIB / avg_row_size / 5)
        assert actual == expected

    def test_prepare_data(self):
        with spark_session('test_prepare_data') as spark:
            df = create_xor_data(spark)

            train_rows = df.count()
            schema_cols = ['features', 'y']
            metadata = util._get_metadata(df)
            assert metadata['features']['intermediate_format'] == constants.ARRAY

            to_petastorm = util.to_petastorm_fn(schema_cols, metadata)
            modified_df = df.rdd.map(to_petastorm).toDF()
            data = modified_df.collect()

            prepare_data = remote._prepare_data_fn(metadata)
            features = torch.tensor([data[i].features for i in range(train_rows)])
            features_prepared = prepare_data('features', features)
            assert np.array_equal(features_prepared, features)

    def test_torch_param_serialize(self):
        serialized_backend = _torch_param_serialize(EstimatorParams.backend.name, 'dummy_value')
        assert serialized_backend is None

        serialized_store = _torch_param_serialize(EstimatorParams.store.name, 'dummy_value')
        assert serialized_store is None

        serialized_dummy_param = _torch_param_serialize('dummy_param_name', None)
        assert serialized_dummy_param is None

    def test_direct_parquet_train(self):
        with spark_session('test_direct_parquet_train') as spark:
            df = create_noisy_xor_data(spark)

            backend = CallbackBackend()
            with local_store() as store:
                store.get_train_data_path = lambda v=None: store._train_path
                store.get_val_data_path = lambda v=None: store._val_path

                with util.prepare_data(backend.num_processes(),
                                       store,
                                       df,
                                       feature_columns=['features'],
                                       label_columns=['y'],
                                       validation=0.2):
                    model = create_xor_model()

                    est = hvd.TorchEstimator(
                        backend=backend,
                        store=store,
                        model=model,
                        input_shapes=[[2]],
                        feature_cols=['features'],
                        label_cols=['y'],
                        validation=0.2,
                        batch_size=1,
                        epochs=2,
                        verbose=2)

                    transformer = est.fit_on_parquet()
                    predictions = transformer.transform(df)
                    assert predictions.count() == df.count()

    def test_legacy_calculate_loss_with_sample_weight(self):
        labels = torch.tensor([[1.0, 2.0, 3.0]])
        outputs = torch.tensor([[1.0, 0.0, 2.0]])

        def fn_minus(output, label, reduction=None):
            losses = label-output
            if reduction == 'none':
                return losses
            else:
                return losses.mean()

        def fn_add(output, label, reduction=None):
            losses = label+output
            if reduction == 'none':
                return losses
            else:
                return losses.mean()

        kwargs = dict(model=mock.Mock(), optimizer=mock.Mock(), feature_cols=[], sample_weights_col='', validation=0)
        model = to_lightning_module(loss_fns=[fn_minus], loss_weights=[1], label_cols=['a'],  **kwargs)
        loss = model._calculate_loss(outputs, labels, sample_weights=torch.tensor([1.0, 6.0, 3.0]))
        assert loss == 5.0

        labels = torch.tensor([[1.0, 2.0, 3.0], [0.0, 2.0, 4.0]])
        outputs = torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 2.0]])

        model = to_lightning_module(loss_fns=[fn_minus, fn_add], loss_weights=[0.2, 0.8], label_cols=['a', 'b'], **kwargs)
        loss = model._calculate_loss(outputs, labels, sample_weights=torch.tensor([1.0, 6.0, 3.0]))
        assert loss == torch.tensor(9.0)

    def test_legacy_calculate_loss_without_sample_weight(self):
        labels = torch.tensor([[1.0, 2.0, 3.0]])
        outputs = torch.tensor([[1.0, 0.0, 2.0]])

        def fn_minus(output, label, reduction=None):
            losses = label-output
            if reduction == 'none':
                return losses
            else:
                return losses.mean()

        def fn_add(output, label, reduction=None):
            losses = label+output
            if reduction == 'none':
                return losses
            else:
                return losses.mean()

        kwargs = dict(model=mock.Mock(), optimizer=mock.Mock(), feature_cols=[], sample_weights_col=None, validation=0)
        model = to_lightning_module(loss_fns=[fn_minus], loss_weights=[1], label_cols=['a'],  **kwargs)
        loss = model._calculate_loss(outputs, labels)
        assert loss == 1.0

        labels = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]])
        outputs = torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 2.0]])

        model = to_lightning_module(loss_fns=[fn_minus, fn_add], loss_weights=[0.2, 0.8], label_cols=['a', 'b'], **kwargs)
        loss = model._calculate_loss(outputs, labels)
        assert torch.isclose(loss, torch.tensor(2.6))
