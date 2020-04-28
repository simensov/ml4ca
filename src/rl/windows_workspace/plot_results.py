from spinup.utils.plot import make_plots

if __name__ == '__main__':
    '''
    # Run single plot as:
    # python .\plot_results.py 'data\secondtest\' --value AverageEpRet AverageVVals StdEpRet --smooth 10
    # Run several plots on top of each other as
    # python .\plot_results.py '.\data\0403_simple_baseline' '.\data\0405_simple_epoch1pt5_3x32layers' '.\data\0406_simple_leakyrelu_2.0epoch_1.2perepch' --value AverageEpRet LossV --smooth 50 --xaxis Time --legend Baseline MoreTrain Leaky
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--value', '-y', default='AverageEpRet', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    parser.add_argument('--add', default=False)
    args = parser.parse_args()

    make_plots(args.logdir, args.legend, args.xaxis, args.value, args.count, 
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est,add=args.add)