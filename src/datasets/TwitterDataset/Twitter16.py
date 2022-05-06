from .TwitterDataset import TwitterDataset


class Twitter16(TwitterDataset):
    def __init__(self, root, tokenizer, max_token, 
                max_tree, batch_size, 
                nfold, determ, num_workers, seed):
        super().__init__(root, 'twitter16', tokenizer, 
                        max_token, max_tree, batch_size, 
                        nfold, determ, num_workers, seed)
        