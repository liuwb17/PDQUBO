import time
from jax.experimental import sparse
import jax
import jax.numpy as jnp
import optax
from jax import jit, grad, lax
from typing import Tuple, Optional, Callable
import numpy as np

import problem_parser


class PDQUBO_JAX:
    def __init__(
            self,
            n_vars: int,
            Q_indices: np.ndarray,
            Q_values: np.ndarray,
            c: jnp.ndarray,
            optimizer_type: str = 'rmsprop',
            batch_size: int = 1,
            primal_lr: float = 0.001,
            tolerance: float = 1e-8,
            dual_lr: float = 0.001,
            dual_init: float = 4,
            max_iters: int = 9999,
            seed: int = 0,
            verbose: bool = True,

    ):

        assert optimizer_type in {'rmsprop', 'adam'}, "Invalid optimizer type"


        self.key = jax.random.PRNGKey(seed)

        self.n = n_vars
        self.m = Q_indices.shape[1]
        self.batch_size = batch_size
        self.Q_indices = Q_indices
        self.Q_values = Q_values
        self.Q = sparse.BCOO((Q_values, jnp.column_stack(Q_indices)), shape=(n_vars, n_vars))
        self.c = c
        self.tolerance = tolerance
        self.primal_lr = primal_lr
        self.dual_lr = dual_lr
        self.max_iters = max_iters
        self.verbose = verbose

        self.primal, self.dual = self._init_variables(dual_init)
        self.incumbent = jnp.full(self.n, 0, dtype=jnp.float32)
        self.objVal = jnp.array(0., dtype=jnp.float32)
        self.objVal_record = [0.]
        self.timing_record = [0.]
        self.start_time = None
        self.solving_time = None

        self.optimizer_primal = self._configure_optimizer(optimizer_type, primal_lr)
        self.optimizer_dual = self._configure_optimizer(optimizer_type, dual_lr)
        self.opt_state_primal = self.optimizer_primal.init(self.primal)
        self.opt_state_dual = self.optimizer_dual.init(self.dual)
        self.log = self._log_verbose if self.verbose else lambda *args, **kwargs: None

    def _log_verbose(self, t, step):
        print(self.objVal, f'time:{t}')

    def _init_variables(self, dual_init: float) -> Tuple[
        jnp.ndarray, jnp.ndarray]:
        key, subkey = jax.random.split(self.key)
        shape = (self.batch_size, self.n)
        primal = jax.random.uniform(subkey, shape)
        dual = jnp.full((self.batch_size, self.n), dual_init, dtype=jnp.float32)
        return primal, dual

    def _configure_optimizer(self, optimizer_type: str, lr: float) -> optax.GradientTransformation:
        if optimizer_type == 'rmsprop':
            return optax.rmsprop(lr, decay=0.98, eps=1e-8, momentum=0.91)
        return optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8)

    def optimize(self):
        self.start_time = time.perf_counter()

        def obj(x):
            # x [n]
            # return (self.Q_values * x[self.Q_indices[0]] * x[self.Q_indices[1]]).sum() + jnp.dot(self.c, x)
            return (((x @ self.Q) + self.c) * x).sum()
        batch_obj = jax.vmap(obj, in_axes=0)

        def base_term(x):
            return batch_obj(x).sum()

        def penalty_term(x, y):
            return (y * (x ** 2 - x)).sum()


        grad_x_fn = jax.grad(lambda x, y: base_term(x) + penalty_term(x, y), argnums=0)

        @jax.jit
        def primal_dual_update(x: jnp.ndarray, y: jnp.ndarray, opt_state_x, opt_state_y, objVal, incumbent):
            grad_x = grad_x_fn(x, y)
            updates_x, opt_state_x = self.optimizer_primal.update(grad_x, opt_state_x, x)
            x = optax.apply_updates(x, updates_x)
            x = jnp.clip(x, 0.0, 1.0)
            y += self.dual_lr * (x ** 2 - x)

            ### Update incumbent
            int_x = jax.lax.stop_gradient(jnp.round(x))
            int_x = jnp.concatenate((int_x, incumbent[jnp.newaxis, :]), axis=0)
            objs = batch_obj(int_x)
            idx = jnp.argmin(objs)
            objVal = objs[idx]
            incumbent = int_x[idx]
            return x, y, opt_state_x, opt_state_y, objVal, incumbent

        for _step in range(self.max_iters):
            t = time.perf_counter()
            (self.primal, self.dual, self.opt_state_primal, self.opt_state_dual, self.objVal,
             self.incumbent) = primal_dual_update(
                self.primal, self.dual, self.opt_state_primal, self.opt_state_dual, self.objVal, self.incumbent)

            self.primal.block_until_ready()
            self.dual.block_until_ready()
            self.incumbent.block_until_ready()

            self.objVal.block_until_ready()
            if self.objVal < self.objVal_record[-1]:
                self.objVal_record.append(self.objVal.item())
                self.timing_record.append(time.perf_counter() - self.start_time)
            self.log(time.perf_counter() - t, _step)

        self.solving_time = time.perf_counter() - self.start_time


class MAX_K_CUT_JAX:

    def __init__(
            self,
            n_vars: int,
            Q_indices: np.ndarray,
            Q_values: np.ndarray,
            c: jnp.ndarray,
            optimizer_type: str = 'rmsprop',
            batch_size: int = 1,
            primal_lr: float = 0.001,
            dual_lr: float = 0.001,
            dual_init: float = 4,
            max_iters: int = 10000,
            seed: int = 0,
            k: int = 3,
            verbose: bool = True
    ):

        assert optimizer_type in {'rmsprop', 'adam'}, "Invalid optimizer type"

        self.key = jax.random.PRNGKey(seed)
        self.n = n_vars
        self.m = Q_indices.shape[1]
        self.batch_size = batch_size
        self.Q_indices = Q_indices
        self.Q_values = Q_values
        self.Q = sparse.BCOO((Q_values, jnp.column_stack(Q_indices)), shape=(n_vars, n_vars))
        self.Q_sum = self.Q.sum()
        self.k = k
        self.c = c
        self.primal_lr = primal_lr
        self.dual_lr = dual_lr
        self.max_iters = max_iters
        self.verbose = verbose

        self.primal, self.dual = self._init_variables(dual_init)
        incumbent = jnp.full((self.k - 1, self.n), 0, dtype=jnp.float32)
        incumbet_last_row = jnp.full((1, self.n), 1, dtype=jnp.float32)
        self.incumbent = jnp.concatenate((incumbent, incumbet_last_row), axis=0)

        self.objVal = jnp.array(0.)
        self.objVal_record = [0.]
        self.timing_record = [0.]
        self.start_time = None
        self.solving_time = None


        self.optimizer_primal = self._configure_optimizer(optimizer_type, primal_lr)
        self.optimizer_dual = self._configure_optimizer(optimizer_type, dual_lr)
        self.opt_state_primal = self.optimizer_primal.init(self.primal)
        self.opt_state_dual = self.optimizer_dual.init(self.dual)
        self.cur_lag = jnp.array(0.)
        self.log = self._log_verbose if self.verbose else lambda *args, **kwargs: None

    def _log_verbose(self, t):
        print(self.objVal, f'time:{t}')

    def _init_variables(self, dual_init: float) -> Tuple[
        jnp.ndarray, jnp.ndarray]:
        key, subkey = jax.random.split(self.key)
        shape = (self.batch_size, self.k, self.n)
        primal = jax.random.uniform(subkey, shape)
        dual = jnp.full((self.batch_size, self.n), dual_init, dtype=jnp.float32)

        return primal, dual

    def _configure_optimizer(self, optimizer_type: str, lr: float) -> optax.GradientTransformation:

        if optimizer_type == 'rmsprop':
            return optax.rmsprop(lr, decay=0.98, eps=1e-8, momentum=0.91)
        return optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8)



    def optimize(self):
        self.start_time = time.perf_counter()


        def obj(x):
            # x [k, n]
            return (jnp.trace(x @ self.Q @ x.T) - self.Q_sum) / 2


        batch_obj = jax.vmap(obj, in_axes=0)

        def base_term(x):
            return batch_obj(x).sum()
        def penalty_term(x, y):
            #  p[b, k, n] //  y [b, n]
            return (y * ((x ** 2).sum(1) - 1)).sum()

        grad_x_fn = jax.grad(lambda x, y: base_term(norm_vmap(x)) + penalty_term(norm_vmap(x), y), argnums=0)
        norm_vmap = jax.vmap(jax.vmap(lambda x: jnp.abs(x) / jnp.sum(jnp.abs(x)), (-1), -1), (0,), 0)


        @jax.jit
        def primal_dual_update(x, y, opt_state_x, opt_state_y, best_obj, incumbent):
            grad_x = grad_x_fn(x, y)
            updates_x, opt_state_x = self.optimizer_primal.update(grad_x, opt_state_x, x)
            x = optax.apply_updates(x, updates_x)
            x = jnp.clip(x, 0.0)
            p = norm_vmap(x)
            y += self.dual_lr * ((p ** 2).sum(1) - 1)

            one_hot_indices = jnp.argmax(x, axis=1)
            int_x = jax.lax.stop_gradient(jax.nn.one_hot(one_hot_indices, x.shape[1], axis=1))  # int_x [b, k, n]
            int_x = jnp.concatenate((int_x, incumbent[jnp.newaxis, :, :]), axis=0)

            objs = batch_obj(int_x)
            idx = jnp.argmin(objs)
            best_obj = objs[idx]
            incumbent = int_x[idx]

            return x, y, opt_state_x, opt_state_y, incumbent, best_obj

        for _step in range(self.max_iters):
            t = time.perf_counter()

            (self.primal, self.dual, self.opt_state_primal, self.opt_state_dual, self.incumbent, self.objVal) = primal_dual_update(
                x=self.primal, y=self.dual, opt_state_x=self.opt_state_primal,
                opt_state_y=self.opt_state_dual, best_obj=self.objVal, incumbent=self.incumbent
            )
            # print(base_term(self.incumbent[jnp.newaxis, :, :]))

            self.primal.block_until_ready()
            self.dual.block_until_ready()
            self.incumbent.block_until_ready()
            self.cur_lag.block_until_ready()
            self.objVal.block_until_ready()
            if self.objVal < self.objVal_record[-1]:
                self.objVal_record.append(self.objVal.item())
                self.timing_record.append(time.perf_counter() - self.start_time)
            self.log(time.perf_counter() - t)
        self.solving_time = time.perf_counter() - self.start_time



class MAXSAT_JAX:
    def __init__(
            self,
            n_vars: int,
            CNF: np.ndarray,
            optimizer_type: str = 'rmsprop',
            batch_size: int = 1,
            primal_lr: float = 0.001,
            dual_lr: float = 0.001,
            dual_init: float = 4,
            max_iters: int = 9999,
            seed: int = 0,
            verbose: bool = True,
    ):

        assert optimizer_type in {'rmsprop', 'adam'}, "Invalid optimizer type"


        self.key = jax.random.PRNGKey(seed)

        self.num_vars = n_vars
        self.num_clause = CNF.shape[0]
        self.batch_size = batch_size
        self.indices = np.abs(CNF) - 1
        self.sign = np.sign(CNF)
        self.primal_lr = primal_lr
        self.dual_lr = dual_lr
        self.max_iters = max_iters
        self.verbose = verbose

        self.primal, self.dual = self._init_variables(dual_init)
        self.incumbent = jnp.full(self.num_vars, 0, dtype=jnp.float32)

        self.objVal = jnp.prod(jnp.where(self.sign < 0,
                                         self.incumbent[self.indices],
                                         1 - self.incumbent[self.indices]), axis=1).sum()


        self.objVal_record = [self.objVal.item()]
        self.timing_record = [0.]
        self.start_time = None
        self.solving_time = None

        self.optimizer_primal = self._configure_optimizer(optimizer_type, primal_lr)
        self.optimizer_dual = self._configure_optimizer(optimizer_type, dual_lr)
        self.opt_state_primal = self.optimizer_primal.init(self.primal)
        self.opt_state_dual = self.optimizer_dual.init(self.dual)
        self.log = self._log_verbose if self.verbose else lambda *args, **kwargs: None

    def _log_verbose(self, t):
        # print(self.objVal, f'time:{t}, integrality:{(self.primal - self.primal ** 2).mean()}, dual:{self.dual.mean()}')
        print(self.objVal)
    def _init_variables(self, dual_init: float) -> Tuple[
        jnp.ndarray, jnp.ndarray]:

        key, subkey = jax.random.split(self.key)

        shape = (self.batch_size, self.num_vars)

        primal = jax.random.uniform(subkey, shape)

        dual = jnp.full((self.batch_size, self.num_vars), dual_init, dtype=jnp.float32)

        return primal, dual

    def _configure_optimizer(self, optimizer_type: str, lr: float) -> optax.GradientTransformation:

        if optimizer_type == 'rmsprop':
            return optax.rmsprop(lr, decay=0.98, eps=1e-8, momentum=0.91)
        return optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8)

    def optimize(self):
        self.start_time = time.perf_counter()

        def obj(x):
            # The number of UNsatisfied clauses
            x_literal = jnp.where(self.sign < 0,
                                  x[self.indices],
                                  1 - x[self.indices])  # shape [m, k]
            clause_values = jnp.prod(x_literal, axis=1)  # shape [m]
            return jnp.sum(clause_values)

        batch_obj = jax.vmap(obj, in_axes=0)

        def base_term(x):
            # x [b, n]
            return batch_obj(x).sum()

        def _g(x):
            # return x**2-x
            return x*jnp.log(x)+(1-x)*jnp.log(1-x)
            
        def penalty_term(x, y):
            return (y * (_g(x))).sum()

        grad_x_fn = jax.grad(lambda x, y: base_term(x) + penalty_term(x, y), argnums=0)

        @jax.jit
        def primal_dual_update(x: jnp.ndarray, y: jnp.ndarray, opt_state_x, opt_state_y, objVal, incumbent):
            grad_x = grad_x_fn(x, y)
            updates_x, opt_state_x = self.optimizer_primal.update(grad_x, opt_state_x, x)
            x = optax.apply_updates(x, updates_x)

            x = jnp.clip(x, 1e-4, 1-1e-4)
            y += self.dual_lr * _g(x)

            ### Update incumbent
            int_x = jax.lax.stop_gradient(jnp.round(x))
            int_x = jnp.concatenate((int_x, incumbent[jnp.newaxis, :]), axis=0)
            objs = batch_obj(int_x)
            # print(objs.shape)
            # print(objs)
            idx = jnp.argmin(objs)
            objVal = objs[idx]
            incumbent = int_x[idx]

            return x, y, opt_state_x, opt_state_y, objVal, incumbent

        for _step in range(self.max_iters):
            t = time.perf_counter()
            (self.primal, self.dual, self.opt_state_primal, self.opt_state_dual, self.objVal,
             self.incumbent) = primal_dual_update(
                self.primal, self.dual, self.opt_state_primal, self.opt_state_dual, self.objVal, self.incumbent)

            self.primal.block_until_ready()
            self.dual.block_until_ready()
            self.incumbent.block_until_ready()
            self.objVal.block_until_ready()
            if self.objVal < self.objVal_record[-1]:
                self.objVal_record.append(self.objVal.item())
                self.timing_record.append(time.perf_counter() - self.start_time)
            self.log(time.perf_counter() - t)

        self.solving_time = time.perf_counter() - self.start_time




