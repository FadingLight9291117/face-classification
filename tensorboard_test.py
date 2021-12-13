from torch.utils.tensorboard import SummaryWriter


tb_writer = SummaryWriter('./tb_test')

loss = 1

iter_num = 0

tb_writer.add_scalar('loss', loss, iter_num, walltime=None)
