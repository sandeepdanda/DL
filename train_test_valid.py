import torch

def training_testing_and_validation(model,criterion,optimizer,num_epochs,train_dataloader,test_dataloader,valid_dataloader):

  # Lists to store the average loss and accuracy for each epoch
  train_loss_history = []
  val_loss_history = []
  train_accuracy_history = []
  val_accuracy_history = []

  # Training
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_inputs, batch_labels in train_dataloader:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs)
        # batch_labels = batch_labels.view(-1, 1) #Resize labels to have the same shape as model outputs
        # Calculate the loss
        # loss = criterion(outputs, batch_labels)
        loss = criterion(outputs, batch_labels.unsqueeze(1))
        # Calculate the training accuracy
        predictions = (outputs > 0.5).float()
        correct_train += (predictions == batch_labels.unsqueeze(1)).sum().item()
        total_train += batch_labels.size(0)


        # Backprop and optmization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Calculate the average training loss and accuracy for the epoch
    average_train_loss = total_loss / len(train_dataloader)
    train_accuracy = correct_train / total_train

    # Validation
    model.eval()
    total_correct = 0
    total_samples = 0
    total_val_loss = 0.0

    with torch.no_grad():
        for batch_inputs, batch_labels in valid_dataloader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)


            outputs = model(batch_inputs)
            # batch_labels = batch_labels.view(-1, 1)

            # val_loss = criterion(outputs, batch_labels)
            val_loss = criterion(outputs, batch_labels.unsqueeze(1))

            predictions = (outputs > 0.5).float()
            total_correct += (predictions == batch_labels.unsqueeze(1)).sum().item()
            total_samples += batch_labels.size(0)
            total_val_loss += val_loss.item()

    average_val_loss = total_val_loss / len(valid_dataloader)
    val_accuracy = total_correct / total_samples

    # Store the values in history
    train_loss_history.append(average_train_loss)
    val_loss_history.append(average_val_loss)
    train_accuracy_history.append(train_accuracy)
    val_accuracy_history.append(val_accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs},")
    print(f"Training Loss: {average_train_loss:.4f},")
    print(f"Training accuracy: {train_accuracy * 100:.2f}%,")
    print(f"Validation Loss: {average_val_loss:.4f},")
    print(f"Validation acurracy: {val_accuracy * 100:.2f}%")
    
    
  # Validation
  model.eval()
  total_correct = 0
  total_samples = 0

  with torch.no_grad():
    for batch_inputs, batch_labels in test_dataloader:
      batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
      outputs = model(batch_inputs)
      predictions = (outputs > 0.5).float()
      total_correct += (predictions == batch_labels.unsqueeze(1)).sum().item()
      total_samples += batch_labels.size(0)

      test_accuracy = total_correct / total_samples

  print(f"Test Acurracy: {test_accuracy * 100:.2f}%")
  
  
