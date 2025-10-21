
def import_model(args):

    print("=> creating model {}".format(args.model_name))
    # create model
    if args.model_name == 'ours':
        from model.nips24.ours import Net
        args.loss = 'reltoabs2'
        model = Net(args)
    return model