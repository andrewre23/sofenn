import pickle

import numpy
import pytest
from keras import losses as losses_module
from keras import ops as k
from keras.src import backend
from keras.src import dtype_policies
from keras.src import testing

from sofenn.losses import CustomLoss


class CustomLossTest(testing.TestCase):
    def setUp(self):
        self._global_dtype_policy = dtype_policies.dtype_policy.dtype_policy()
        self._floatx = backend.floatx()
        return super().setUp()

    def tearDown(self):
        dtype_policies.dtype_policy.set_dtype_policy(self._global_dtype_policy)
        backend.set_floatx(self._floatx)
        return super().tearDown()

    def test_reduction(self):
        y_true = numpy.array([1.0, 0.0, 1.0, 0.0])
        y_pred = numpy.array([0.1, 0.2, 0.3, 0.4])

        manual_loss = 0.5 * (y_pred - y_true) ** 2

        # # No reduction
        loss_fn = CustomLoss(reduction=None)
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(manual_loss, loss)

        # sum
        loss_fn = CustomLoss(reduction="sum")
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(numpy.sum(manual_loss), loss)

        # sum_over_batch_size or mean
        loss_fn = CustomLoss(reduction="sum_over_batch_size")
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(numpy.sum(manual_loss) / 4, loss)

        # bad reduction
        with self.assertRaisesRegex(ValueError, "Invalid value for argument"):
            CustomLoss(reduction="abc")

    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="Numpy backend does not support masking.",
    )
    def test_mask(self):
        mask = numpy.array([True, False, True, True])
        y_true = numpy.array([1.0, 0.0, 1.0, 0.0])
        y_pred = numpy.array([0.1, 0.2, 0.3, 0.4])

        masked_y_true = numpy.array([1.0, 1.0, 0.0])
        masked_y_pred = numpy.array([0.1, 0.3, 0.4])

        mask = k.convert_to_tensor(mask)
        y_true = k.convert_to_tensor(y_true)
        y_pred = k.convert_to_tensor(y_pred)
        y_pred._keras_mask = mask

        loss_fn = CustomLoss()
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(
            numpy.sum(0.5 * (masked_y_true - masked_y_pred) ** 2), loss
        )

        # Test edge case where everything is masked.
        mask = numpy.array([False, False, False, False])
        y_pred._keras_mask = mask
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(loss, 0)  # No NaN.

    def test_sample_weight(self):
        sample_weight = numpy.array([0.4, 0.3, 0.2, 0.1])
        y_true = numpy.array([1.0, 0.0, 1.0, 0.0])
        y_pred = numpy.array([0.1, 0.2, 0.3, 0.4])

        loss_fn = CustomLoss()
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(
            numpy.sum(sample_weight * 0.5 * (y_pred - y_true) ** 2), loss
        )

        # Test edge case where every weight is 0.
        sample_weight = numpy.array([0.0, 0.0, 0.0, 0.0])
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(loss, 0)  # No NaN.

    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="Numpy backend does not support masking.",
    )
    def test_mask_and_sample_weight(self):
        sample_weight = numpy.array([0.4, 0.3, 0.2, 0.1])
        y_true = numpy.array([1.0, 0.0, 1.0, 0.0])
        y_pred = numpy.array([0.1, 0.2, 0.3, 0.4])
        mask = numpy.array([True, False, True, True])

        masked_sample_weight = numpy.array([0.4, 0.2, 0.1])
        masked_y_true = numpy.array([1.0, 1.0, 0.0])
        masked_y_pred = numpy.array([0.1, 0.3, 0.4])

        mask = k.convert_to_tensor(mask)
        y_true = k.convert_to_tensor(y_true)
        y_pred = k.convert_to_tensor(y_pred)
        y_pred._keras_mask = mask

        loss_fn = CustomLoss()
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(
            numpy.sum(masked_sample_weight * 0.5 * (masked_y_true - masked_y_pred) ** 2),
            loss,
        )

    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="Numpy backend does not support masking.",
    )
    def test_mask_and_sample_weight_rank2(self):
        # check loss of inputs with duplicate rows doesn't change
        sample_weight = numpy.array([0.4, 0.3, 0.2, 0.1])
        y_true = numpy.array([1.0, 0.0, 1.0, 0.0])
        y_pred = numpy.array([0.1, 0.2, 0.3, 0.4])
        mask = numpy.array([True, False, True, True])

        mask = k.convert_to_tensor(mask)
        y_true = k.convert_to_tensor(y_true)
        y_pred = k.convert_to_tensor(y_pred)
        y_pred._keras_mask = mask

        loss_fn = CustomLoss(reduction="sum_over_batch_size")
        rank1_loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)

        # duplicate rows
        mask = k.tile(k.expand_dims(mask, axis=0), (2, 1))
        y_true = k.tile(k.expand_dims(y_true, axis=0), (2, 1))
        y_pred = k.tile(k.expand_dims(y_pred, axis=0), (2, 1))
        sample_weight = k.tile(k.expand_dims(sample_weight, axis=0), (2, 1))
        y_pred._keras_mask = mask
        rank2_loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(rank1_loss, rank2_loss)

    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="Numpy backend does not support masking.",
    )
    def test_rank_adjustment(self):
        for uprank in ["mask", "sample_weight", "ys"]:
            sample_weight = numpy.array([0.4, 0.3, 0.2, 0.1])
            y_true = numpy.array([1.0, 0.0, 1.0, 0.0])
            y_pred = numpy.array([0.1, 0.2, 0.3, 0.4])
            mask = numpy.array([True, False, True, True])

            if uprank == "mask":
                mask = numpy.expand_dims(mask, -1)
            elif uprank == "sample_weight":
                sample_weight = numpy.expand_dims(sample_weight, -1)
            elif uprank == "ys":
                y_true = numpy.expand_dims(y_true, -1)
                y_pred = numpy.expand_dims(y_pred, -1)

            masked_sample_weight = numpy.array([0.4, 0.2, 0.1])
            masked_y_true = numpy.array([1.0, 1.0, 0.0])
            masked_y_pred = numpy.array([0.1, 0.3, 0.4])

            mask = k.convert_to_tensor(mask)
            y_true = k.convert_to_tensor(y_true)
            y_pred = k.convert_to_tensor(y_pred)
            y_pred._keras_mask = mask

            loss_fn = CustomLoss()
            loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
            self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
            self.assertAllClose(
                numpy.sum(masked_sample_weight * 0.5 * (masked_y_true - masked_y_pred) ** 2),
                loss,
            )

    def test_mixed_dtypes(self):
        sample_weight = numpy.array([0.4, 0.3, 0.2, 0.1], dtype="float64")
        y_true = numpy.array([1.0, 0.0, 1.0, 0.0], dtype="int32")
        y_pred = numpy.array([0.1, 0.2, 0.3, 0.4], dtype="float32")

        loss_fn = CustomLoss()
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(
            numpy.sum(sample_weight * 0.5 * (y_pred - y_true) ** 2),
            loss,
        )

    def test_pickle(self):
        loss = losses_module.get("mse")
        loss = pickle.loads(pickle.dumps(loss))
        self.assertEqual(loss, losses_module.mean_squared_error)

    def test_get_method(self):
        loss = losses_module.get("mse")
        self.assertEqual(loss, losses_module.mean_squared_error)

        loss = losses_module.get(None)
        self.assertEqual(loss, None)

        with self.assertRaises(ValueError):
            losses_module.get("typo")

    def test_dtype_arg(self):
        y_true = numpy.array([1.0, 0.0, 1.0, 0.0], dtype="float32")
        y_pred = numpy.array([0.1, 0.2, 0.3, 0.4], dtype="float32")

        # Note: we use float16 and not float64 to test this because
        # JAX will map float64 to float32.
        loss_fn = CustomLoss(dtype="float16")
        loss = loss_fn(y_true, y_pred)
        self.assertDType(loss, "float16")

        # Test DTypePolicy for `dtype` argument
        loss_fn = CustomLoss(dtype=dtype_policies.DTypePolicy("mixed_float16"))
        loss = loss_fn(y_true, y_pred)
        self.assertDType(loss, "float16")

        # `dtype` setter should raise AttributeError
        with self.assertRaises(AttributeError):
            loss_fn.dtype = "bfloat16"

    def test_default_dtype(self):
        y_true = numpy.array([1.0, 0.0, 1.0, 0.0], dtype="float32")
        y_pred = numpy.array([0.1, 0.2, 0.3, 0.4], dtype="float32")

        # Defaults to `keras.config.floatx()` not global `dtype_policy`
        dtype_policies.dtype_policy.set_dtype_policy("mixed_float16")
        loss_fn = CustomLoss()
        loss = loss_fn(y_true, y_pred)
        self.assertDType(loss, "float32")

        backend.set_floatx("float16")
        loss_fn = CustomLoss()
        loss = loss_fn(y_true, y_pred)
        self.assertDType(loss, backend.floatx())

    def test_from_config(self):
        y_true = numpy.random.random(3)
        y_pred = numpy.random.random(3)

        loss_fn = CustomLoss(reduction="sum")
        self.assertTrue(loss_fn(y_true, y_pred) == CustomLoss.from_config(loss_fn.get_config())(y_true, y_pred))
