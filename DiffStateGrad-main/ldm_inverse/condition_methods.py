from abc import ABC, abstractmethod
import torch
import numpy as np

__CONDITIONING_METHOD__ = {}


def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls

    return wrapper


def get_conditioning_method(name: str, model, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](model=model, operator=operator, noiser=noiser, **kwargs)

class ConditioningMethod(ABC):
    def __init__(self, model , operator, noiser, **kwargs):
        self.model = model
        self.operator = operator
        self.noiser = noiser

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)

    def grad_and_value(self, x_prev, x_0_hat, measurement, sigma, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(self.model.differentiable_decode_first_stage( x_0_hat ), **kwargs)
            norm = torch.linalg.norm(difference)
            # todo what is sigma sigma is sqrt variance
            # print(sigma)
            # print(norm.shape)
            # norm_likelihood = (norm) / (2 * sigma ** 2)
            norm_likelihood = norm

            norm_grad_likelihood = (-1) * \
                                   torch.autograd.grad(outputs=norm_likelihood, inputs=x_prev)[0]
            # norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev, retain_graph=True)[0]#∇xi ∥y − A(xˆ0)∥2没求平方

            # fixme
            norm_grad = 0

        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
            # fixme
            norm_grad_likelihood = 0

        else:
            raise NotImplementedError

        # return norm_grad, norm, norm_grad_likelihood
        return norm_grad, norm, norm_grad_likelihood

    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass


@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, model,operator, noiser, **kwargs):
        super().__init__(model,operator, noiser)
        self.operator = operator
        self.scale = kwargs.get('scale', 1.0)

    def calculateS(self, prior_score, likelihood_score):  # CFGzero源代码
        """
            s = ((prior + likelihood) · prior) / ||prior||^2

            Inputs:
                - prior_score: (B, C, H, W)
                - likelihood_score: (B, C, H, W)

            Returns:
                - st_star: (B, 1, 1, 1) for broadcasting
            """
        B = prior_score.shape[0]

        prior_flat = prior_score.view(B, -1)
        combined_flat = (prior_score + likelihood_score).view(B, -1)

        dot_product = torch.sum(combined_flat * prior_flat, dim=1, keepdim=True)  # (B, 1)
        squared_norm = torch.sum(prior_flat ** 2, dim=1, keepdim=True) + 1e-8  # (B, 1)

        st_star = dot_product / squared_norm  # (B, 1)

        return st_star.view(B, 1, 1, 1)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, alpha_bar, noise_prior_score, sample_step,sigma, alphat,
                     **kwargs):
        # todo: norm_likelihood是负值
        # print(sigma)
        norm_grad, norm, norm_grad_likelihood = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat,
                                                                    measurement=measurement,
                                                                    sigma=sigma, **kwargs)
        #
        # print(f"alpha_bar shape: {alpha_bar.shape}")
        # print(f"noise_prior_score shape: {noise_prior_score.shape}")

        # prior_score = (-1) * (1 / np.sqrt(1 - alpha_bar)) * noise_prior_score
        # 将numpy的sqrt换成torch.sqrt
        prior_score = (-1) * (1 / torch.sqrt(1 - alpha_bar)) * noise_prior_score

        #todo
        s = self.calculateS(prior_score, norm_grad_likelihood)


        # print(f"[conditioning] step={sample_step}, s={s}")

        # s =1
        # print(s)
        # print(f"alphat type: {type(alphat)}")
        Ourlambda = self.scale * torch.sqrt(alphat) * (sigma ** 2) / ((1 - alphat) * norm)
        # x_t += (self.scale * norm_grad_likelihood * 2 * norm * (1 - alphat) / torch.sqrt(alphat) + (s - 1) * (1 - Ourlambda) * prior_score)
        x_t += (self.scale * norm_grad_likelihood + (s - 1) * (1 - Ourlambda) * (1 - alphat) / torch.sqrt(
            alphat) * prior_score)

        return x_t, s  # x_t -= norm_grad * self.scale#xi−1 ← x′  i−1 − ζi∇xi ∥y − A(xˆ0)∥2  2
