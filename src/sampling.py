from .predictor_euler_maruyama import EulerMaruyamaPredictor
from .ode_sampler import ODESampler


def get_prediction(
        x,
        t,
        score,
        device,
        eps,
        sde_model,
        sde_solver='ode',
        score_fn= None,
):
   
    if sde_solver == 'euler_maruyama':
        predictor = EulerMaruyamaPredictor(
            sde=sde_model,
            score_fn=score_fn,
            probability_flow=True)

    elif sde_solver == 'ode':
        predictor = ODESampler(
            sde_model=sde_model,
            device=device,
            eps=eps)

    if sde_solver == 'euler_maruyama':
        x_hat, x_mean = predictor.update_fn(x, t, score)
    elif sde_solver == 'ode':
        x_hat, _ = predictor.update_fn(x, t, score)
    return x_hat