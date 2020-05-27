# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm
from tvm import te

import numpy as np

def test_hoist_relaxed_if():
    N = te.placeholder((), name='N', dtype="int32")
    i = te.reduce_axis((0, N()))
    A = te.compute((), lambda: te.sum(i * 2, axis=i), name='A')
    dtype = A.dtype

    s = te.create_schedule(A.op)
    i_axis, = A.op.reduce_axis
    i_outer, i_axis = s[A].split(i_axis, 4)

    mod = tvm.lower(s, [N, A], name="f")
    assert(type(mod["f"].body.seq[1]) is tvm.tir.stmt.For)
    assert(type(mod["f"].body.seq[1].body) is tvm.tir.stmt.IfThenElse)
    assert(type(mod["f"].body.seq[1].body.then_case) is tvm.tir.stmt.For)
    assert(type(mod["f"].body.seq[1].body.then_case.body) is not tvm.tir.stmt.IfThenElse)
    assert(type(mod["f"].body.seq[1].body.else_case) is tvm.tir.stmt.For)
    assert(type(mod["f"].body.seq[1].body.else_case.body) is tvm.tir.stmt.IfThenElse)

    ctx = tvm.cpu(0)
    mod = tvm.build(mod, [A], target="llvm")
    N_nd = tvm.nd.array(np.full((), 15, dtype=dtype), ctx)
    A_nd = tvm.nd.array(np.zeros((), dtype=dtype), ctx)
    mod(N_nd, A_nd)
    tvm.testing.assert_allclose(A_nd.asnumpy(), np.full((), 210))

if __name__ == "__main__":
    test_hoist_relaxed_if()
