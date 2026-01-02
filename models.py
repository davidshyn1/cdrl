import copy
import torch
from torch import nn

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        
        # obs_space가 Dict space인지 확인하고 shapes 추출
        import gym.spaces as spaces
        
        try:
            # 1. Dict space인 경우
            if isinstance(obs_space, spaces.Dict):
                shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
            # 2. Box space인 경우
            elif isinstance(obs_space, spaces.Box):
                shapes = {'observations': tuple(obs_space.shape)}
            # 3. spaces 속성이 있는 경우 (Dict-like)
            elif hasattr(obs_space, 'spaces'):
                spaces_dict = obs_space.spaces
                # spaces가 속성인지 메서드인지 확인
                if callable(spaces_dict):
                    spaces_dict = spaces_dict()
                # dict-like 객체에서 shapes 추출
                if hasattr(spaces_dict, 'items'):
                    shapes = {k: tuple(v.shape) if hasattr(v, 'shape') else tuple(v) 
                             for k, v in spaces_dict.items()}
                else:
                    raise ValueError(f"obs_space.spaces is not dict-like: {type(spaces_dict)}")
            # 4. shape 속성이 직접 있는 경우
            elif hasattr(obs_space, 'shape'):
                shapes = {'observations': tuple(obs_space.shape)}
            else:
                # 지원하지 않는 타입
                raise ValueError(f"Unsupported observation space type: {type(obs_space)}")
        except ValueError:
            # ValueError는 그대로 전파
            raise
        except Exception as e:
            # 기타 예외는 상세 정보와 함께 전파
            raise ValueError(
                f"Error extracting shapes from observation space (type: {type(obs_space)}): {str(e)}"
            ) from e
        ### Observation Encoder q(x_t|o_t)
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        ### Dynamics Model q(s_t|s_{t-1}, a_{t-1})
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete: # if the dynamics model is discrete
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        ### Decoder p(o_t|s_t)
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        ### Reward Head p(r_t|s_t,a_t)
        self.heads["rewards"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            rewards=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _train(self, data, eval_mode=False):
        # action (batch_size, batch_length, act_dim)
        # observations (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        data = self.preprocess(data)

        if not eval_mode:
            # 명시적으로 모든 서브모듈을 train 모드로 설정
            self.train(True)
            self.encoder.train(True)
            self.dynamics.train(True)
            for head in self.heads.values():
                head.train(True)
        else:
            self.eval()
            self.encoder.eval()
            self.dynamics.eval()
            for head in self.heads.values():
                head.eval()

        with tools.RequiresGrad(self):         
            with torch.amp.autocast('cuda:1', enabled=self._use_amp):
                # encoder 실행
                embed = self.encoder(data["observations"])
                

                post, prior = self.dynamics.observe(
                    embed, data["actions"], data["is_first"]
                )

                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )

                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
                
                # torch.mean을 autocast 블록 안에서 호출하여 gradient 유지
                mean_loss = torch.mean(model_loss)
                
                # optimizer 호출도 autocast 블록 안에서 수행 (gradient 그래프 유지)
                metrics = self._model_opt(mean_loss, self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.amp.autocast('cuda:1', enabled=self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        # torch.tensor()는 새로운 텐서를 생성하므로 gradient를 끊을 수 있음
        # 이미 텐서인 경우 .to()를 사용하고, numpy array인 경우만 torch.tensor() 사용
        processed_obs = {}
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                # 이미 텐서인 경우 device와 dtype만 변경 (gradient 유지)
                processed_obs[k] = v.to(device=self._config.device, dtype=torch.float32)
            else:
                # numpy array 등인 경우 새 텐서 생성 (입력 데이터이므로 gradient 불필요)
                processed_obs[k] = torch.tensor(v, device=self._config.device, dtype=torch.float32)
        obs = processed_obs
        obs["observations"] = obs["observations"] / 255.0
        if "discounts" in obs:
            obs["discounts"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discounts"] = obs["discounts"] #.unsqueeze(-1)
        obs["cont"] = (1.0 - obs["terminals"]) #.unsqueeze(-1)
        
        # is_first 처리: 없으면 생성, 있으면 shape 확인 후 변환
        if "is_first" not in obs:
            batch_size, batch_length = obs["observations"].shape[:2]
            is_first = torch.zeros((batch_size, batch_length), device=self._config.device, dtype=torch.float32)
            is_first[:, 0] = 1.0  # 각 batch의 첫 번째 step만 True
            obs["is_first"] = is_first
        else:
            # is_first가 있으면 float32로 변환하고 shape 확인
            is_first = obs["is_first"].float()
            # shape가 (batch_size, batch_length)가 아니면 reshape
            if is_first.ndim == 1:
                # 1D인 경우 (batch_size * batch_length,) -> (batch_size, batch_length)로 reshape
                batch_size, batch_length = obs["observations"].shape[:2]
                is_first = is_first.view(batch_size, batch_length)
            elif is_first.ndim > 2:
                # 3D 이상인 경우 마지막 차원 제거
                is_first = is_first.squeeze(-1)
            obs["is_first"] = is_first
        
        return obs

    # ##TODO : Check this function when video prediction is implemented
    # def video_pred(self, data):
    #     data = self.preprocess(data)
    #     embed = self.encoder(data)

    #     states, _ = self.dynamics.observe(
    #         embed[:6, :5], data["s"][:6, :5], data["is_first"][:6, :5]
    #     )
    #     recon = self.heads["decoder"](self.dynamics.get_feat(states))["observations"].mode()[
    #         :6
    #     ]
    #     reward_post = self.heads["rewards"](self.dynamics.get_feat(states)).mode()[:6]
    #     init = {k: v[:, -1] for k, v in states.items()}
    #     prior = self.dynamics.imagine_with_action(data["actions"][:6, 5:], init)
    #     openl = self.heads["decoder"](self.dynamics.get_feat(prior))["observations"].mode()
    #     reward_prior = self.heads["rewards"](self.dynamics.get_feat(prior)).mode()
    #     # observed image is given until 5 steps
    #     model = torch.cat([recon[:, :5], openl], 1)
    #     truth = data["observations"][:6]
    #     model = model
    #     error = (model - truth + 1.0) / 2.0

    #     return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective,
    ):
        self._update_slow_target()
        metrics = {}


        # # 모든 파라미터에 requires_grad=True 명시적으로 설정
        # for param in self.actor.parameters():
        #     param.requires_grad = True

        with tools.RequiresGrad(self.actor):
            with torch.amp.autocast('cuda:1', enabled=self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        # actor 컨텍스트가 끝난 후 value 파라미터가 여전히 requires_grad=True인지 확인
        # # 모든 파라미터에 requires_grad=True 명시적으로 설정 (actor 컨텍스트의 영향 제거)
        # for param in self.value.parameters():
        #     if not param.requires_grad:
        #         print(f"WARNING: value param requires_grad=False before RequiresGrad context, fixing...")
        #     param.requires_grad = True

        with tools.RequiresGrad(self.value):
            with torch.amp.autocast('cuda:1', enabled=self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
