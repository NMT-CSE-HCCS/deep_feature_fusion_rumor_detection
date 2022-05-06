from .TwitterDataset import TwitterDataset


class Twitter15(TwitterDataset):
    def __init__(self, root, tokenizer, max_token, 
                max_tree, batch_size, 
                nfold, determ, num_workers, seed):
        super().__init__(root, 'twitter15', tokenizer, 
                        max_token, max_tree, batch_size, 
                        nfold, determ, num_workers, seed)
        