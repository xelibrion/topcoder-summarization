import time

from tqdm import tqdm
import torch

from nnet.meters import AverageMeter


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DummyLRateScheduler(object):
    """
    :param optimizer:  optimizer to tweak learning rate of
    :param steps_completed: steps completed so far across all iterations
    """

    def update(self, optimizer, steps_completed):
        pass


class LearningRateScheduler(object):
    """
    :param schedule:  schedule (e.g. WarmupLinearSchedule) to get learning rate from
    :param learning_rate: default learning rate
    """

    def __init__(self, schedule, learning_rate):
        self.schedule = schedule
        self.learning_rate = learning_rate

    """
    :param optimizer:  optimizer to tweak learning rate of
    :param steps_completed: steps completed so far across all iterations
    """

    def update(self, optimizer, steps_completed):
        lr_this_step = self.learning_rate * self.schedule.get_lr(steps_completed)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_step


class TrainigLoop(object):
    def __init__(self, model, optimizer, lr_scheduler, callbacks):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.callbacks = callbacks
        self.epoch_steps = 1000
        self.total_steps_completed = 0

    def run(self, train_iterator, val_iterator, criterion, epochs=30):
        best_val_loss = float('inf')

        for epoch in range(epochs):

            start_time = time.time()

            train_loss = self.train_epoch(train_iterator, criterion)
            val_loss = self.evaluate(val_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.callbacks['on_best_val_loss'](self.model)

            print(f'\nEpoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {val_loss:.3f}\n')

    def train_epoch(self, iterator, criterion):

        loss_meter = AverageMeter(int(len(iterator) / 10))
        self.model.train()

        epoch_loss = 0

        num_steps = self.epoch_steps or len(iterator)
        steps_completed = 0

        with tqdm(total=num_steps) as tq:
            tq.set_description('Train')

            for batch in iterator:
                self.optimizer.zero_grad()

                for idx, t in enumerate(batch):
                    batch[idx] = t.to(device)

                input_ids, attention_mask, segments_ids, cls_ids, cls_mask, labels = batch
                # labels = [batch_size x 512]

                output = self.model(input_ids, attention_mask, segments_ids, cls_ids,
                                    cls_mask)
                sent_scores, out_cls_mask = output

                # sent_scores = [batch_size x 512]
                # out_cls_mask = [batch_size x 512]

                loss = criterion(sent_scores, labels)
                loss = (loss * out_cls_mask.float()).sum()
                (loss / loss.numel()).backward()

                self.optimizer.step()

                loss_meter.update(loss.item())
                tq.set_postfix(loss='{:.3f}'.format(loss_meter.mavg), )
                tq.update()

                epoch_loss += loss.item()
                steps_completed += 1
                self.total_steps_completed += 1
                self.lr_scheduler.update(self.optimizer, self.total_steps_completed)

                if steps_completed >= num_steps:
                    break

        return epoch_loss / steps_completed

    def evaluate(self, iterator, criterion):

        self.model.eval()
        loss_meter = AverageMeter(int(len(iterator) / 10))

        epoch_loss = 0

        num_steps = self.epoch_steps or len(iterator)
        steps_completed = 0

        with tqdm(total=num_steps) as tq:
            tq.set_description('Validation')

            with torch.no_grad():

                for batch in iterator:
                    for idx, t in enumerate(batch):
                        batch[idx] = t.to(device)

                    input_ids, attention_mask, segments_ids, cls_ids, cls_mask, labels = batch
                    # labels = [batch_size x 512]

                    output = self.model(input_ids, attention_mask, segments_ids, cls_ids,
                                        cls_mask)
                    sent_scores, out_cls_mask = output

                    loss = criterion(sent_scores, labels)
                    loss = (loss * out_cls_mask.float()).sum()

                    epoch_loss += loss.item()
                    loss_meter.update(loss.item())
                    tq.set_postfix(loss='{:.3f}'.format(loss_meter.mavg), )
                    tq.update()

                    steps_completed += 1
                    if steps_completed >= num_steps:
                        break

        return epoch_loss / steps_completed
