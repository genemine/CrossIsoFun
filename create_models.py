def create_model(opt,num_feature, num_label, dim_list, dim_he_list, dim_hvcdn, device):
    model = None
    print(opt.model)
    if opt.model == 'vigan':
        from VIGAN_model import VIGANModel
        assert(opt.align_data == False)
        model = VIGANModel()
    model.initialize(opt, num_feature, num_label, dim_list, dim_he_list, dim_hvcdn, device)
    print("model [%s] was created" % (model.name()))
    return model
