def print_progress_bar(iteration, total, length):
    percent = ("{0:.2f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = '█' * filledLength + '-' * (length - filledLength)
    print('\rProgress: |%s| %s%% Complete' % (bar, percent), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print('\n')