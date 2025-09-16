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


def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError

        return norm_grad, norm

    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass


@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t



@register_conditioning_method(name='dmps')  # DMPS method
class PosteriorSampling_meng(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        self.zero_init_steps = kwargs.get('zero_init_steps', 10)

    def calculateS(self, prior_score, likelihood_score):
        """
            s = ((prior + posterior) · prior) / ||prior||^2

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

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, H_funcs, noise_std, alpha_t, alpha_bar, pseudonoise_scale, noise_prior_score, sample_step, total_step, mask=None, **kwargs):

        singulars = H_funcs.singulars()
        S = singulars * singulars.to(x_t.device)
        alpha_bar = np.clip(alpha_bar, 1e-16, 1 - 1e-16)
        alpha_t = np.clip(alpha_t, 1e-16, 1 - 1e-16)
        scale_S = (1 - alpha_bar) / (alpha_bar)
        S_vector = (1 / (S * scale_S + noise_std ** 2)).to(x_t.device).reshape(-1, 1)
        Temp_value = H_funcs.Ut(measurement - H_funcs.H(x_t) / np.sqrt(alpha_bar)).t()
        grad_value = H_funcs.Ht(H_funcs.U((S_vector * Temp_value).t()))
        grad_value = grad_value.reshape(x_t.shape) / np.sqrt(alpha_bar)
        prior_score = (-1) * (1 / np.sqrt(1 - alpha_bar)) * noise_prior_score

        s = self.calculateS(prior_score, grad_value)
        # s = 1 #if s = 1 go back to origin dmps
        x_t += (1 - alpha_t) / np.sqrt(alpha_t) * (self.scale * grad_value + (s - 1) * (1 - self.scale) * prior_score)


        return x_t

@register_conditioning_method(name='pgdm')
class PosteriorSampling_meng(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)


    def calculateS(self, prior_score, likelihood_score):
        """
            s = ((prior + posterior) · prior) / ||prior||^2

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

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, H_funcs, noise_std, alpha_t, alpha_bar_prev, alpha_bar, noise_prior_score, pseudonoise_scale,  **kwargs):
        # def conditioning(self, x_t, x_0_hat, measurement, H_funcs, alpha_t, alpha_bar, noise_std, noise_prior_score,
        #                  **kwargs):  # pgdm
        singulars = H_funcs.singulars()
        S = singulars*singulars.to(x_t.device)
        alpha_bar = np.clip(alpha_bar,1e-16,1-1e-16)
        alpha_t = np.clip(alpha_t,1e-16,1-1e-16)
        #scale_S = (1-alpha_bar)/(alpha_bar)
        scale_S = (1-alpha_bar)/(1+alpha_bar)
        x_t_hat = (x_t - noise_prior_score) / (2.0 - alpha_bar)
         #print(S.shape())
        #print(pseudonoise_scale)
        #S_vector = (1/(S*scale_S*pseudonoise_scale +noise_std**2)).to(x_t.device).reshape(-1,1)
        S_vector = (1/(S*scale_S +noise_std**2)).to(x_t.device).reshape(-1,1)
        S_vector = torch.sqrt(S_vector)
        #Temp_value = H_funcs.Ut(measurement - H_funcs.H(x_t)/np.sqrt(alpha_bar)).t()
        # x_t_hat = np.sqrt(alpha_bar)*x_0_hat +
        # Temp_value = H_funcs.Ut(measurement - H_funcs.H(x_t_hat)).t()
        Temp_value = H_funcs.Ut(measurement - H_funcs.H(x_0_hat)).t()
        grad_norm = -0.5*torch.linalg.norm(S_vector*Temp_value)**2

        #diag = torch.diag(1/(S*scale_S+noise_std**2)).to(x_t.device)
        #grad_value = H_funcs.Ht(H_funcs.U(torch.matmul(diag,H_funcs.Ut(measurement - H_funcs.H(x_t)/np.sqrt(alpha_bar)).t()).t()))

        grad_value_x0 = torch.autograd.grad(outputs=grad_norm, inputs=x_prev)[0]
        # grad_value_x0 = torch.autograd.grad(outputs=grad_norm, inputs=x_t_hat)[0]
        #print(grad_value_x0.shape) #print(grad_value_x0.shape())
        #print(grad_value.max())
        #print(grad_value.min())

        prior_score = (-1) * (1 / np.sqrt(1 - alpha_bar)) * noise_prior_score

        s = self.calculateS(prior_score, grad_value_x0)
        # s=1
        x_t = x_t + (1 - alpha_t) / np.sqrt(alpha_t) * (self.scale * grad_value_x0 + (s - 1) * (1 - self.scale) * prior_score)

        # x_t += self.scale * grad_value_x0 *(1-alpha_t)/np.sqrt(alpha_t) #/100

        #x_t += pseudonoise_scale*grad_value *(1-alpha_t)/np.sqrt(alpha_t) #/100
        #x_t += grad_value*np.sqrt(alpha_bar_prev)/np.sqrt(alpha_bar)*(1-alpha_t)/np.sqrt(1-alpha_bar)

        #print(x_t.max())
        #print(x_t.min())

        return x_t