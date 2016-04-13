from nltk.tokenize import word_tokenize

def load_data(dataset):
    "This function loads the dataset"
    program_start_token = "PROGRAM_START"
    program_end_token = "PROGRAM_END"

    # Read the data and append PROGRAM_START and PROGRAM_END tokens
    print "Reading progs file..."
    with open(dataset + '_progs.txt', 'rb') as f:
        progs = f.readlines()
        # Append PROGRAM_START and PROGRAM_END
        progs = ["%s %s %s" % (program_start_token, prog, program_end_token) for prog in progs]
        # Split program into tokens
        progs = [word_tokenize(prog) for prog in progs]
    
    # Read the labels for correctness of the program
    with open(dataset + '_labels.txt', 'rb') as f:
        labels = [bool(int(l)) for l in f]

    num_progs = len(progs)
    # Make sure number of programs and labels match
    if num_progs != len(labels):
        print "Incorrect input data"
    else:
        print "Parsed %d progs." % num_progs
    return progs, labels, num_progs

def load_char_data(dataset):
    "This function loads the dataset"

    # Read the data and append PROGRAM_START and PROGRAM_END tokens
    print "Reading progs file..."
    with open(dataset + '_progs.txt', 'rb') as f:
        progs = f.readlines()
        # Split program into tokens
        progs = [ list(prog) for prog in progs]
        progs = [prog[:-2] for prog in progs]

    # Read the labels for correctness of the program
    with open(dataset + '_labels.txt', 'rb') as f:
        labels = f.readlines()
        labels = [list(label) for label in labels]
        labels = [label[:-1] for label in labels]
        labels = [[int(c) if int(c)!=2 else 0.5 for c in label] for label in labels]
        #labels = [bool(int(l)) for l in f]

    num_progs = len(progs)
    # Make sure number of programs and labels match
    if num_progs != len(labels):
        print "Incorrect input data"
    else:
        print "Parsed %d progs." % num_progs
    return progs, labels, num_progs

def parse_args(argv, training_dataset, testing_dataset, batch_size, num_epochs, verbose):
    "This function is common parsing for command line arguments"

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--training_dataset', '-tr', nargs='?', default=training_dataset,
        help='Location of training dataset'
    )
    parser.add_argument(
        '--testing_dataset', '-te', nargs='?', default=testing_dataset,
        help='Location of testing dataset'
    )
    parser.add_argument(
        '--batch_size', '-b', nargs='?', type=int, default=batch_size,
        help='Batch size'
    )
    parser.add_argument(
        '--num_epochs', '-e', nargs='?', type=int, default=num_epochs,
        help='Number of epochs'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true', default=verbose, 
        help='arguments to be passed on to the client Jupyter notebook'
    )
    args = parser.parse_args(argv)
    print args

    return args.training_dataset, args.testing_dataset, args.batch_size, args.num_epochs, args.verbose
