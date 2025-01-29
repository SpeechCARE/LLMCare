class SubNet(nn.Module):

    def __init__(self, inter_fusion, input_size, mid_size, SEED):
        super(SubNet, self).__init__()
        self.seed = torch.manual_seed(SEED)
        self.projector = nn.Linear(input_size, mid_size)
        self.projector_to_2 = nn.Linear(mid_size, 2)
        self.inter_fusion = inter_fusion

    def forward(self, features):
        x = F.relu(self.projector(features))
        if not self.inter_fusion :
            x = self.projector_to_2(x)
        return x


class Network(nn.Module):

    def __init__(self, transformers_name, embedding_output_structure, SEED, extended=False, fusion_type=None, sub_inter_fusion=False, trans_mid_size=256,
                 ling_input_size=32, ling_mid_size=32, liwc_input_size=90, liwc_mid_size=8):

      super(Network, self).__init__()
      self.seed = torch.manual_seed(SEED)
      self.embedding_output_structure = embedding_output_structure
      self.transformers_name = transformers_name
      self.transformer_model = AutoModel.from_pretrained(transformers_name)
      self.extended = extended
      self.fusion_type = fusion_type
      self.sub_inter_fusion = sub_inter_fusion

      classifier_input = 768

      if self.extended:
          self.transformer_projector = nn.Linear(768, trans_mid_size)
          classifier_input = trans_mid_size

      if fusion_type != None:
          self.transformer_projector_to_2 = nn.Linear(trans_mid_size if self.extended else 768, 2)
          self.ling_net = SubNet(self.sub_inter_fusion or self.fusion_type=='intermediate', ling_input_size, ling_mid_size, SEED)
          self.liwc_net = SubNet(self.sub_inter_fusion or self.fusion_type=='intermediate', liwc_input_size, liwc_mid_size, SEED)

          if self.sub_inter_fusion:
              # Linear layer to project the concatenation of linguistic and LIWC features into 2 dimensions
              self.all_ling_projector = nn.Linear(ling_mid_size + liwc_mid_size, 2)

          self.trans_weight = nn.Parameter(torch.rand(1))
          self.ling_weight = nn.Parameter(torch.rand(1))
          self.liwc_weight = nn.Parameter(torch.rand(1))
          self.all_ling_weight = nn.Parameter(torch.rand(1))

          if self.fusion_type == 'intermediate':
              if self.extended:
                  classifier_input = trans_mid_size + ling_mid_size + liwc_mid_size
              else:
                  print("\nNote: It's not possible to do intermediate fusion when 'extended' is False\n")
                  sys.exit(0)

          else : # All types of late fusion
              classifier_input = 4 if self.sub_inter_fusion else 6

      self.classifier = nn.Linear(classifier_input, 2)


    def freeze_all(self):
        for param in self.transformer_model.parameters():
            param.requires_grad = False


    def finetune_all(self):
        for param in self.transformer_model.parameters():
            param.requires_grad = True


    def finetune_n_last_layers(self, last_n_layers, all_layers):
        error_flag = 0
        for i, model_child in enumerate(self.transformer_model.children()):
            if i!=1:
                for param in model_child.parameters():
                    param.requires_grad = False
            else:
                if last_n_layers == all_layers:
                    print("Please set 'fine-tuning' variabel to 'All layers'.")
                    error_flag = 1
                    break
                elif last_n_layers > all_layers:
                    print("Number of all layers in the {} are {}!".format(self.transformers_name, all_layers))
                    error_flag = 1
                    break
                print('All layers:', all_layers)
                last_n_layers_list = np.arange(1, last_n_layers+1)
                for num_layer, module in enumerate(model_child.children()):

                    if self.transformers_name == 'xlnet-base-cased':
                        if num_layer not in (all_layers - last_n_layers_list):
                            print("This layer is freezed: ", num_layer)
                            for param in module.parameters():
                                param.requires_grad = False
                    else:
                        for num_layer, child in enumerate(module.children()):
                            if num_layer not in (all_layers - last_n_layers_list):
                                print("This layer is freezed: ", num_layer)
                                for param in child.parameters():
                                    param.requires_grad = False

        return error_flag


    def fusion_late1(self, x_trans, x_ling, x_liwc, sub_inter_fusion):
        if sub_inter_fusion:
            x_all_ling = self.all_ling_projector(torch.cat((x_ling, x_liwc), dim=1))
            return self.classifier(torch.cat((x_trans, x_all_ling), dim=1))
        else:
            return self.classifier(torch.cat((x_trans, x_ling, x_liwc), dim=1))


    def fusion_late2(self, x_trans, x_ling, x_liwc, sub_inter_fusion):
        if sub_inter_fusion:
            x_all_ling = self.all_ling_projector(torch.cat((x_ling, x_liwc), dim=1))
            return x_trans + x_all_ling
        else:
            return x_trans + x_ling + x_liwc


    def fusion_late3(self, x_trans, x_ling, x_liwc, sub_inter_fusion):
        if sub_inter_fusion:
            x_all_ling = self.all_ling_projector(torch.cat((x_ling, x_liwc), dim=1))
            return (self.trans_weight * x_trans) + (self.all_ling_weight * x_all_ling)
        else:
            return (self.trans_weight * x_trans) + (self.ling_weight * x_ling) + (self.liwc_weight * x_liwc)


    def forward(self, sent_id, mask , ling_features, liwc_features):

        output = self.transformer_model(sent_id, attention_mask=mask)
        # CLS
        if self.embedding_output_structure=='CLS':
            x_transformer = output.last_hidden_state[:, 0, :]
        # Mean
        elif self.embedding_output_structure=='Mean':
            x_transformer = torch.mean(output.last_hidden_state, dim=1)

        if self.extended:
            x_transformer = F.tanh(self.transformer_projector(x_transformer))

        if self.fusion_type == None:
            output = self.classifier(x_transformer)
        else:
            x_ling = self.ling_net(ling_features)
            x_liwc = self.liwc_net(liwc_features)

            if self.fusion_type == 'intermediate':
                output = self.classifier(torch.cat((x_transformer, x_ling, x_liwc), dim=1))

            else: # All types of late fusion
                x_transformer = self.transformer_projector_to_2(x_transformer)

                if self.fusion_type == 'late1':
                    output = self.fusion_late1(x_transformer, x_ling, x_liwc, self.sub_inter_fusion)
                elif self.fusion_type == 'late2':
                    output = self.fusion_late2(x_transformer, x_ling, x_liwc, self.sub_inter_fusion)
                elif self.fusion_type == 'late3':
                    output = self.fusion_late3(x_transformer, x_ling, x_liwc, self.sub_inter_fusion)

        return output
