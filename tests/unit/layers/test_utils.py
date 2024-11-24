from sofenn.layers.utils import get_fuzzy_output_shape

def test_output_shape():
    test_cases = {
        ((3,),      5):      (5,),
        ((3, 7),    5):    (3, 5),
        ((3, 7, 2), 5): (3, 7, 5),
    }
    for (input_shape, neurons), output_shape in test_cases.items():
        assert get_fuzzy_output_shape(input_shape=input_shape, neurons=neurons) == output_shape