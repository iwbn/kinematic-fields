from omegaconf import OmegaConf

def update_model_param(cfg, dataset):
    # set OmegaConf modifiable
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.model, False)

    if isinstance(dataset, dict):
        keys = dataset.keys()
    else:
        keys = dataset.__dict__

    # modify config for scene flow
    if dataset.ndc_ray:
        cfg['model']['ndc_params'] = {'H': float(dataset.img_wh[1]), 
                            'W': float(dataset.img_wh[0]),
                            'focal': float(dataset.focal[0]),
                            'near': float(1.0), 'use_ndc': True,}
        if 'center' in keys:
            cfg['model']['ndc_params']['center'] = [
                float(dataset.center[0]), float(dataset.center[1])]
        else:
            cfg['model']['ndc_params']['center'] = [
                float(dataset.img_wh[0]/2.), float(dataset.img_wh[1]/2.)]
        
        cfg['model']['ndc_system'] = True
    else:
        cfg['model']['ndc_params'] = {'H': float(dataset.img_wh[1]), 
                            'W': float(dataset.img_wh[0]),
                            'focal': float(dataset.focal[0]),
                            'near': float(1.0), 'use_ndc': False,}
        if 'center' in keys:
            cfg['model']['ndc_params']['center'] = [
                float(dataset.center[0]), float(dataset.center[1])]
        else:
            cfg['model']['ndc_params']['center'] = [
                float(dataset.img_wh[0]/2.), float(dataset.img_wh[1]/2.)]
            
        cfg['model']['ndc_system'] = False
                
    cfg['model']['time_interval'] = dataset.frame_interval

    # set OmegaConf not modifiable
    OmegaConf.set_struct(cfg.model, True)
    OmegaConf.set_struct(cfg, True)
    