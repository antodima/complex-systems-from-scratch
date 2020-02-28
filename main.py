#!/usr/bin/env python3
import numpy as np

from complex_systems import discrete_complex_systems as discrete
from complex_systems import continuous_complex_systems as continuous


linear_model = discrete.LinearModel(rd=1.5, sd=0.9, beta=10)
linear_model.evolve(initial_population=20, t_max=20)
#print('LinearModel evolution at time', 5, linear_model.evolution_at_t(5))
#print('LinearModel equilibrium:', linear_model.equilibrium())
#linear_model.plot()

non_linear_model = discrete.NonLinearModel(rd=2, k=50)
non_linear_model.evolve(initial_population=20, t_max=20)
#non_linear_model.plot()

ac_model = discrete.ACModel(alpha=2.5, beta=3)
ac_model.evolve(initial_a_population=20, initial_c_population=5, max_time=20)
#ac_model.plot()

# initial condition
y0 = [1, 1, 2]
# time points
t = np.linspace(0, 2, 120000)
oregonator_model = continuous.OregonatorModel(t)
oregonator_model.evolve(y0)
#oregonator_model.plot()

y0 = [800, 1000]
#y0 = [200, 200]
t = np.linspace(0, 4, 2000)
lv_model = continuous.LotkaVolterraModel(t)
lv_model.evolve(y0)
#lv_model.plot()

y0 = [0.99, 0.01, 0]
t = np.linspace(0, 10, 10)
sir_model =  continuous.SIRModel(t, infection_coefficient=1800, 
                                 vaccination_rate=0.5,
                                 recovery_rate=100,
                                 growth_rate=0.02)
sir_model.evolve(y0)
#sir_model.plot()
#print('Vaccination threshold', sir_model.vaccination_threshold())

y0 = [800, 1000, 5]
#y0 = [200, 200]
t = np.linspace(0, 4, 2000)
ppv_model = continuous.PPVModel(t)
ppv_model.evolve(y0)
#ppv_model.plot()

