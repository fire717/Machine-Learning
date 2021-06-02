def myCELoss(self, pre, label):
        #print(pre.shape, label.shape)#torch.Size([2764, 3]) torch.Size([2764]

        ### 原始CE loss
        #loss = F.cross_entropy(pre, label, reduction='sum') #e0 loss 7.9068

        ### CE loss等价实现1
        # log_soft_out = F.log_softmax(pre, dim=-1)
        # loss = F.nll_loss(log_soft_out, label, reduction='sum')

        ### CE loss等价实现2
        # soft_out = F.softmax(pre, dim=-1)
        # log_soft_out = torch.log(soft_out)
        # loss = F.nll_loss(log_soft_out, label, reduction='sum')

        ### CE loss等价实现3
        # log_soft_out = F.log_softmax(pre, dim=-1)
        # one_hot = F.one_hot(label, pre.shape[1]).float().to(self.device)
        # loss = torch.sum(-one_hot * log_soft_out)

        ### label smooth
        log_soft_out = F.log_softmax(pre, dim=-1)
        one_hot = F.one_hot(label, pre.shape[1]).float().to(pre.device)
        one_hot = one_hot * (1-self.labelsmooth)+self.labelsmooth/pre.shape[1]
        loss = torch.sum(-one_hot * log_soft_out)

        ### label smooth, 加强face when==facemask
        # log_soft_out = F.log_softmax(pre, dim=-1)
        # one_hot = F.one_hot(label, pre.shape[1]).float().to(self.device)
        # one_hot = one_hot * (1-self.labelsmooth)+self.labelsmooth/pre.shape[1]
        # facemask_index = label==2
        # one_hot[facemask_index,1] = one_hot[facemask_index,0]+one_hot[facemask_index,1]
        # one_hot[facemask_index,2] = one_hot[facemask_index,2]-one_hot[facemask_index,0]
        # loss = torch.sum(-one_hot * log_soft_out)


        # print(loss) #4388.9595/1.5879
        # #b
        return loss
