/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file hoist_and_split_if.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/arith/analyzer.h>
#include <vector>

namespace tvm {
namespace tir {

class HoistRelaxedIfMutator : public StmtMutator {
 protected:
  Stmt VisitStmt_(const ForNode *op) override {
    op = StmtMutator::VisitStmt_(op).as<ForNode>();

    // condition of each IfThenElse nodes, from outer to inner
    std::vector<PrimExpr> conds;
    Stmt body = op->body;
    while (auto op = body.as<IfThenElseNode>()) {
      // we don't hoist 2-branch IfThenElse nodes
      if (op->else_case.defined()) {
        break;
      }
      conds.push_back(op->condition);
      body = op->then_case;
    }

    Stmt ret = ForNode::make(op->loop_var, op->min, op->extent,
        op->for_type, op->device_api, body);
    auto op_end = AddNode::make(op->min, op->extent);

    int name_cnt = 0;
    // iterate from inner to outer
    for (auto it = conds.rbegin(); it != conds.rend(); it++) {
      PrimExpr cond_relaxed = Relax_(*it, op->loop_var, op->min, op_end);
      body = IfThenElseNode::make(*it, body);
      Var new_var(op->loop_var->name_hint + std::to_string(name_cnt++));
      Stmt new_body = Substitute(Stmt(CopyOnWrite(body.as<IfThenElseNode>())),
          Map<Var, PrimExpr>{{op->loop_var, new_var}});
      auto slow_path = ForNode::make(new_var, op->min, op->extent,
          op->for_type, op->device_api, new_body);
      if (cond_relaxed.defined()) {
        ret = IfThenElseNode::make(cond_relaxed, ret, slow_path);
      } else {  // always else_case
        ret = slow_path;
      }
    }
    return ret;
  }

 private:
  PrimExpr Relax_(const PrimExpr &cond, const Var &var,
      const PrimExpr &begin, const PrimExpr &end) {
    PrimExpr expr;  // reduce to expr < 0 or expr <= 0
    if (auto op = cond.as<LTNode>()) {
      expr = SubNode::make(op->a, op->b);
    } else if (auto op = cond.as<LENode>()) {
      expr = SubNode::make(op->a, op->b);
    } else if (auto op = cond.as<GTNode>()) {
      expr = SubNode::make(op->b, op->a);
    } else if (auto op = cond.as<GENode>()) {
      expr = SubNode::make(op->b, op->a);
    } else {
      return PrimExpr();
    }

    arith::Analyzer analyzer;
    analyzer.Bind(var, Range(begin, end));
    Map<Var, PrimExpr> to_min{{std::make_pair(var, begin)}};
    if (analyzer.CanProveGreaterEqual(
          SubNode::make(Substitute(expr, to_min), expr), 0)) {
      return Substitute(cond, to_min);
    }
    Map<Var, PrimExpr> to_max{{std::make_pair(var, SubNode::make(end, 1))}};
    if (analyzer.CanProveGreaterEqual(
          SubNode::make(Substitute(expr, to_max), expr), 0)) {
      return Substitute(cond, to_max);
    }
    return PrimExpr();
  }
};


namespace transform {

Pass HoistRelaxedIf() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = HoistRelaxedIfMutator()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.HoistRelaxedIf", {});
}

TVM_REGISTER_GLOBAL("tir.transform.HoistRelaxedIf")
.set_body_typed(HoistRelaxedIf);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
