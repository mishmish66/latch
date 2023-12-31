from latch.infos import Infos

import jax
from jax import numpy as jnp

from jax.experimental.host_callback import barrier_wait


def test_infos_basics_host():
    infos = Infos()
    infos = infos.add_info("foo", 1)

    assert infos.infos == {"foo": 1}, "Infos object should contain the added info."

    infos = infos.add_info("bar", 2)

    assert infos["bar"] == 2, "Indexing infos should give added info."


def test_infos_basics_device():
    infos = Infos()

    def foo(x, infos):
        return jnp.ones((4, 3)) * x, infos.add_info("foo", x)

    ys, infos = jax.vmap(foo)(jnp.arange(5), infos)

    assert jnp.allclose(
        ys, jnp.ones((5, 4, 3)) * jnp.arange(5)[:, None, None]
    ), "Function should return the correct values."

    info_dict = infos.infos
    assert len(info_dict) == 1, "Infos object should contain one info."
    assert "foo" in info_dict, "Infos object should contain the added info."
    assert jnp.allclose(
        info_dict["foo"], jnp.arange(5)
    ), "Infos object should contain the added info."


def test_infos_printout_host():
    infos = Infos()
    infos = infos.add_info("foo", 1)
    infos = infos.add_info("bar", 2)

    # Redirect stdout to a string
    import sys
    from contextlib import contextmanager
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    infos.host_dump_to_console(16)
    result = mystdout.getvalue()
    expected = "Infos for step 16:\n    foo: 1\n    bar: 2\n"

    assert result == expected, "Infos should be printed correctly."

    sys.stdout = mystdout = StringIO()
    infos = infos.add_info("baz", {"a": 1, "b": 2})
    infos.host_dump_to_console(16)
    result = mystdout.getvalue()
    expected = (
        """Infos for step 16:\n    foo: 1\n    bar: 2\n    baz/a: 1\n    baz/b: 2\n"""
    )

    assert result == expected, "Infos should be printed correctly."

    sys.stdout = old_stdout


def test_infos_printout_device():
    infos = Infos()

    def foo(x, infos):
        return jnp.ones((4, 3)) * x, infos.add_info("foo", x)

    ys, infos = jax.vmap(foo)(jnp.arange(5), infos)

    # Redirect stdout to a string
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    jax.jit(Infos.dump_to_console)(infos, 16)
    barrier_wait()
    result = mystdout.getvalue()
    expected = "Infos for step 16:\n    foo: [0 1 2 3 4]\n"

    assert result == expected, "Infos should be printed correctly."

    sys.stdout = old_stdout
