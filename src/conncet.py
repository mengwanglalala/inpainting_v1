import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .models import TeacherModel, StudentModel
from .dataset import Dataset
from .metrics import PSNR
from .utils import Progbar, create_dir, stitch_images, imsave

class Connect():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = 'teacher'
        elif config.MODEL == 2:
            model_name = 'student'
        elif config.MODEL == 3:
            model_name = 'joint'

        self.model_name = model_name
        self.teacher_model = TeacherModel(config).to(config.DEVICE)
        self.student_model = StudentModel(config).to(config.DEVICE)
        self.psnr = PSNR(255.0).to(config.DEVICE)

        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST,
                                        augment=False, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST,
                                         augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST,
                                       augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')



    def save(self):
        if self.config.MODEL == 1:
            self.teacher_model.save()

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.student_model.save()

        else:
            self.teacher_model.save()
            self.student_model.save()

    def load(self):
        if self.config.MODEL == 1:
            self.teacher_model.load()

        elif self.config.MODEL == 2:
            self.teacher_model.load()
            self.student_model.load()

        else:
            self.teacher_model.load()
            self.student_model.load()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while (keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)
            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.teacher_model.train()
                self.student_model.train()

                images, images_gray, edges, masks = self.cuda(*items)

                # edge model
                if model == 1:
                    # train
                    outputs, gen_loss, dis_loss, logs, _ = self.teacher_model.process(images, masks)

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs))
                    mae = (torch.sum(torch.abs(images - outputs)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.teacher_model.backward(gen_loss, dis_loss)
                    iteration = self.teacher_model.iteration

                elif model == 2:
                    # train
                    images_masked = (images * (1 - masks).float()) + masks
                    _, orignal_layer = self.teacher_model(images, masks)
                    _, masked_layer = self.teacher_model(images_masked, masks)
                    # layer_diff = orignal_layer['layer1']- masked_layer['layer1']
                    # print(layer_diff.size())
                    outputs, gen_loss, dis_loss, logs = self.student_model.process(images, masks, orignal_layer, masked_layer)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.student_model.backward(gen_loss, dis_loss)
                    iteration = self.student_model.iteration


                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                           ("epoch", epoch),
                           ("iter", iteration),
                       ] + logs

                progbar.add(len(images),values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])
                #log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

        print('\nEnd training....')


    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        self.teacher_model.eval()
        self.student_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            images, images_gray, edges, masks = self.cuda(*items)

            if model == 1:
                # eval
                outputs, gen_loss, dis_loss, logs, _ = self.teacher_model.process(images, masks)

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs))
                mae = (torch.sum(torch.abs(images - outputs)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

            elif model == 2:
                # eval
                # train
                images_masked = (images * (1 - masks).float()) + masks
                _, orignal_layer = self.teacher_model(images, masks)
                _, masked_layer = self.teacher_model(images_masked, masks)
                # layer_diff = orignal_layer['layer1']- masked_layer['layer1']
                # print(layer_diff.size())
                outputs, gen_loss, dis_loss, logs = self.student_model.process(images, masks, orignal_layer, masked_layer)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

            logs = [("it", iteration), ] + logs
            progbar.add(len(images), values=logs)

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.teacher_model.eval()
        self.student_model.eval()


        model = self.config.MODEL
        items = next(self.sample_iterator)
        images, images_gray, edges, masks = self.cuda(*items)

        # edge model
        if model == 1:
            with torch.no_grad():
                iteration = self.teacher_model.iteration
                inputs = images
                #inputs = (images_gray * (1 - masks)) + masks
                outputs, _ = self.teacher_model(images, masks)
                outputs_merged = (outputs * masks) + (edges * (1 - masks))

        # inpaint model
        elif model == 2:
            with torch.no_grad():
                iteration = self.student_model.iteration
                images_masked = (images * (1 - masks).float()) + masks
                _, orignal_layer = self.teacher_model(images, masks)
                _, masked_layer = self.teacher_model(images_masked, masks)
                inputs = (images * (1 - masks)) + masks
                outputs, gen_loss, dis_loss, logs = self.student_model.process(images, masks, orignal_layer, masked_layer)
                outputs_merged = (outputs * masks) + (images * (1 - masks))


        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        images = stitch_images(
            self.postprocess(images),
            self.postprocess(inputs),
            self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row = image_per_row
        )


        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()