def get_model(args):       
    if args.model.framework == 'UNISURF':
        from .unisurf import get_model
    elif args.model.framework == 'NeuS':
        from .neus import get_model
    elif args.model.framework == 'VolSDF':
        from .volsdf import get_model
    elif args.model.framework == 'PVolSDF':
        from .pvolsdf_sRGB import get_model
    elif args.model.framework == 'PVolSDFMono':
        from .pvolsdf_mono import get_model
    elif args.model.framework == 'PNeuS':
        from .pneus import get_model
    elif args.model.framework == 'SSL-PNeuS':
        from .sslpneus import get_model
    elif args.model.framework == 'SSL-PVolSDF':
        from .sslpvolsdf import get_model
    elif args.model.framework == 'holoNeuS':
        from .holoNeuS import get_model
    elif args.model.framework == 'pnr':
        from .pnr import get_model
    else:
        raise NotImplementedError
    return get_model(args)