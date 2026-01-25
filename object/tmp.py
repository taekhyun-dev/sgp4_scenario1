if event['type'] == 'IOT_TRAIN':
                self.sim_logger.info(f"\n📡 [Time: {event['start_time'].strftime('%m-%d %H:%M')}] SAT_{sat_id} : IoT 학습 시작")
                
                epochs = LOCAL_EPOCHS
                loader_idx = sat_id % len(self.client_subsets)
                dataset = self.client_subsets[loader_idx]

                # [최적화] 고사양 PC에 맞춘 DataLoader 설정
                train_loader = DataLoader(
                        dataset, 
                        batch_size=128,  # 32코어 PC면 128 추천
                        shuffle=True, 
                        num_workers=8,   # 8~16 추천
                        pin_memory=True, 
                        persistent_workers=False 
                )

                current_local_wrapper.to_device(temp_model, device='cpu')

                # [중요] 처음부터 학습(Scratch)이므로 LR을 0.001로 설정 (ImageNet Fine-tuning은 1e-5였음)
                train_model(
                    model=temp_model, 
                    global_state_dict=self.global_model_wrapper.model_state_dict,
                    train_loader=train_loader, 
                    epochs=epochs, 
                    lr=0.005,  # <--- ResNet9 Scratch 학습용 LR (Adam 기본값)
                    device=self.device,
                    sim_logger=self.sim_logger
                )

                # 버전 증가
                next_version = round(current_local_wrapper.version + 0.1, 1)

                acc, loss = self._evaluate_direct(
                    temp_model, 
                    self.val_loader, 
                    sat_id=sat_id, 
                    version=next_version, 
                    stage="LOCAL_TRAIN"
                )
                self.satellite_performances[sat_id] = acc 

                current_local_wrapper = PyTorchModel.from_model(
                    temp_model, 
                    version=next_version
                )
                self.satellite_models[sat_id] = current_local_wrapper

                self.sim_logger.info(f"   ✅ [Result] Acc: {acc:.2f}%, v{current_local_wrapper.version:.1f}")
                