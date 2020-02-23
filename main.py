#!/usr/bin/env python3
from complex_systems import discrate_complex_systems as discrate


linear_model = discrate.LinearModel(rd=1.5, sd=0.9, beta=10)
linear_model.evolve(initial_population=20, t_max=20)
print('LinearModel evolution at time', 5, linear_model.evolution_at_t(5))
print('LinearModel equilibrium:', linear_model.equilibrium())
linear_model.plot()

non_linear_model = discrate.NonLinearModel(rd=2, k=50)
non_linear_model.evolve(initial_population=20, t_max=20)
non_linear_model.plot()

ac_model = discrate.ACModel(alpha=.5, beta=.6)
ac_model.evolve(initial_a_population=50, initial_c_population=5, max_time=20)
ac_model.plot()
