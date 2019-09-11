
class utils:
    def __init__():
        return

    def assert_less(small, big):
        assert small.shape() == big.shape()
        small  = small.data()
        big = big.data()
        for (a,d) in zip(small, big):
            assert (a<d), "ASSERT failed on assert_less"

        return True;

    def assert_equal(actual, desired):
        assert actual.shape() == desired.shape()
        actual  = actual.data()
        desired = desired.data()
        for (a,d) in zip(actual, desired):
            assert (a==d), "ASSERT failed on assert_equal"

        return True;

    def assert_allclose(actual, desired, rtol=1e-07, atol=0):
        assert actual.shape() == desired.shape()
        actual  = actual.data()
        desired = desired.data()
        # actual = atol + rtol * abs(desired)
        for (a,d) in zip(actual, desired):
            assert (abs(a-d) <= atol + rtol * abs(d)), "ASSERT failed on assert_allclose"

        return True;

