from .TwitterDataset import TwitterDataset

class Twitter15(TwitterDataset):
    def __init__(self, tokenizer, max_token, 
                max_tree, batch_size, 
                nfold, determ, num_workers, seed):
        super().__init__('../rumor_detection_acl2017', 'twitter15', tokenizer, 
                        max_token, max_tree, batch_size, 
                        nfold, determ, num_workers, seed)
        