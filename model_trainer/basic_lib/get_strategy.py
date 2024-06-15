import torch
# from lightning.fabric.accelerators import ACCELERATOR_REGISTRY
# from lightning.fabric.strategies import STRATEGY_REGISTRY

from model_trainer.basic_lib.obj_loader import filter_kwargs


def get_strategy(
    devices="auto",
    num_nodes=1,
    accelerator="auto",
    strategy={"name": "auto"},
    precision=None,
):
    from lightning.fabric.utilities.device_parser import _determine_root_gpu_device
    # from lightning.fabric.accelerators import AcceleratorRegistry
    from lightning.fabric.accelerators import ACCELERATOR_REGISTRY
    from lightning.fabric.accelerators.cuda import CUDAAccelerator
    from lightning.fabric.accelerators.mps import MPSAccelerator
    # from lightning.fabric.strategies import  StrategyRegistry
    from lightning.fabric.strategies import STRATEGY_REGISTRY
    from lightning.fabric.strategies import Strategy, SingleDeviceStrategy
    # from lightning.pytorch.trainer.connectors import accelerator_connector
    from lightning.fabric import connector
    from lightning.pytorch.utilities.rank_zero import rank_zero_warn
    class _DsAcceleratorConnector(connector._Connector):
        def __init__(self) -> None:
            # accelerator_connector._register_external_accelerators_and_strategies()
            self._registered_strategies = STRATEGY_REGISTRY.available_strategies()
            self._registered_accelerators = ACCELERATOR_REGISTRY.available_accelerators()
            # super().__init__()
            self._registered_strategies = STRATEGY_REGISTRY.available_strategies()
            self._accelerator_types = ACCELERATOR_REGISTRY.available_accelerators()
            self._parallel_devices = []
            self._check_config_and_set_final_flags(
                strategy=strategy["name"],
                accelerator=accelerator,
                precision=precision,
                plugins=[],
                # sync_batchnorm=False,
            )
            if self._accelerator_flag == "auto":
                self._accelerator_flag = self._choose_auto_accelerator()
            elif self._accelerator_flag == "gpu":
                self._accelerator_flag = self._choose_gpu_accelerator_backend()
            self._check_device_config_and_set_final_flags(devices=devices, num_nodes=num_nodes)
            self._set_parallel_devices_and_init_accelerator()
            if self._strategy_flag == "auto":
                self._strategy_flag = self._choose_strategy()
            self._check_strategy_and_fallback()
            self._init_strategy()
            # for k in ["colossalai", "bagua", "hpu", "hpu_parallel", "hpu_single", "ipu", "ipu_strategy"]:
            #     if k in StrategyRegistry:
            #         StrategyRegistry.remove(k)

        def _init_strategy(self) -> None:
            assert isinstance(self._strategy_flag, (str, Strategy))
            if isinstance(self._strategy_flag, str):
                if self._strategy_flag not in STRATEGY_REGISTRY:
                    available_names = ", ".join(sorted(STRATEGY_REGISTRY.available_strategies())) or "none"
                    raise KeyError(f"Invalid strategy name {strategy['name']}. Available names: {available_names}")
                data = STRATEGY_REGISTRY[self._strategy_flag]
                params = {}
                # Replicate additional logic for _choose_strategy when dealing with single device strategies
                if issubclass(data["strategy"], SingleDeviceStrategy):
                    if self._accelerator_flag == "hpu":
                        params = {"device": torch.device("hpu")}
                    elif self._accelerator_flag == "tpu":
                        params = {"device": self._parallel_devices[0]}
                    elif data["strategy"] is SingleDeviceStrategy:
                        if isinstance(self._accelerator_flag, (CUDAAccelerator, MPSAccelerator)) or (
                            isinstance(self._accelerator_flag, str) and self._accelerator_flag in ("cuda", "gpu", "mps")
                        ):
                            params = {"device": _determine_root_gpu_device(self._parallel_devices)}
                        else:
                            params = {"device": "cpu"}
                    else:
                        raise NotImplementedError
                params.update(data["init_params"])
                params.update({k: v for k, v in strategy.items() if k != "name"})
                self.strategy = data["strategy"](**filter_kwargs(params, data["strategy"]))
            elif isinstance(self._strategy_flag, SingleDeviceStrategy):
                params = {"device": self._strategy_flag.root_device}
                params.update({k: v for k, v in strategy.items() if k != "name"})
                self.strategy = self._strategy_flag.__class__(**filter_kwargs(params, self._strategy_flag.__class__))
            else:
                rank_zero_warn(
                    f"Inferred strategy {self._strategy_flag.__class__.__name__} cannot take custom configurations."
                    f"To use custom configurations, please specify the strategy name explicitly."
                )
                self.strategy = self._strategy_flag

    return _DsAcceleratorConnector().strategy

if __name__=='__main__':
    import lightning as PL
    test=get_strategy(precision='bf16',devices=1,strategy={"name": "ddp"},accelerator="gpu")
    PL.Fabric(

        precision='bf16',

    )
    pass
