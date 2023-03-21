import torch
from copy import deepcopy

EPS = 1e-8

def get_regularizer(model, model_old, device, opts, old_state):
    resume = False
    name = opts.regularizer
    if old_state is not None:
        if name != old_state['name']:
            print(f"Warning: the regularizer passed {name}"
                f"is different from the state one {old_state['name']}")
        resume = True

    if name is None:
        return None

    elif name == 'ewc':
        fisher = old_state["fisher"] if resume else None
        return EWC(model, model_old, device, fisher=fisher,
                    alpha=opts.reg_alpha, normalize=not opts.reg_no_normalize)

    elif name == 'pi':
        score = old_state["score"] if resume else None
        return PI(model, model_old, device, score=score, 
                    normalize=not opts.reg_no_normalize)

    elif name == 'rw':
        score = old_state["score"] if resume else None
        fisher = old_state["fisher"] if resume else None
        return RW(model, model_old, device, score=score, fisher=fisher, 
                    alpha=opts.reg_alpha, iterations=opts.reg_iterations,
                    normalize=not opts.reg_no_normalize)

    else: 
        raise NotImplementedError


def normalize_fn(mat):
    return (mat - mat.min()) / (mat.max() - mat.min() + EPS)


class Regularizer:
    def update(self):
        """ Stub method """
        raise NotImplementedError

    def penalty(self):
        """ Stub method """
        raise NotImplementedError

    def state_dict(self):
        """ Stub method """
        raise NotImplementedError

    def load_state_dict(self, state):
        """ Stub method """
        raise NotImplementedError

class EWC(Regularizer):
    """Regularizer for EWC inheristance from Regularizer Class
    Args:
        model: current training model at task t
        model_old: previous training model at task t-1
        device: device used
        fisher: Fisher Matrix
        alpha: hyper-param for EWC regularizer
        normalize (bool): normalize or not
    """
    def __init__(self, model, model_old, device, fisher=None, alpha=0.9, normalize=True):
        self.model = model
        self.device = device
        self.alpha = alpha
        self.normalize = normalize

        # store model for penalty step
        if model_old is not None:
            self.model_old = model_old
            self.model_old_dict = self.model_old.state_dict()
            self.penalize = True
        else:
            self.penalize = False

        # make fisher matrix for the estimate of parameter important
        # store the old fisher matrix (if exist) for penalize step
        # Initialize the old Fisher Matrix
        if fisher is not None:
            self.fisher_old = fisher
            self.fisher = {}
            for key, par in self.fisher_old.items():
                self.fisher_old[key].requires_grad = False
                self.fisher_old[key] = normalize_fn[par] if normalize else par
                self.fisher_old[key] = self.fisher_old[key].to(device)
                self.fisher[key] = torch.clone(par).to(device)
        # Initialize a new Fisher Matrix and no penalize, miss a information
        else:
            self.fisher_old = None
            self.penalize = False
            self.fisher = {}

        # Update Fisher with new keys (due to incremental classes)
        for n, p in self.model.named_parameters():
            if p.requires_grad and n not in self.fisher:
                self.fisher[n] = torch.ones_like(p, device=device, requires_grad=False)

    def update(self):
        # suppose model have already grad computes, 
        # directly update the fisher by model.paramteres
        for n, p in self.model.named_parameters():
            self.fisher[n] = (self.alpha * (p.grad ** 2)) + ((1 - self.alpha) * self.fisher[n])

    def penalty(self):
        if not self.penalty:
            return 0.
        else:
            loss = 0.
            for n, p in self.model.named_parameters():
                if n in self.model_old_dict and p.requires_grad:
                    loss += (self.fisher_old[n] * (p - self.model_old_dict[n]) ** 2).sum()
            return loss

    def get(self):
        return self.fisher  # return new Fisher Matrix

    def state_dict(self):
        state = {"name": "ewc", "fisher": self.fisher, "alpha": self.alpha,}
        return state

    def load_state_dict(self, state):
        assert state['name'] == 'ewc', f"Error, trying to restore {state['name']} into ewc"
        self.fisher = state["fisher"]
        for k, p in self.fisher.items():
            self.fisher[k] = p.to(self.device)
        self.alpha = state["alpha"]
