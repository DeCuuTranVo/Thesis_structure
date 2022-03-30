import time
if __name__ == "__main__":
    start_time = time.clock()
    # You can change the number of GPUs per trial here:
    main(num_samples=20, max_num_epochs=120, gpus_per_trial=2)
    
    print(time.clock() - start_time, "seconds")