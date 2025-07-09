# dictionary of strategies generated for kyber (modified code from https://github.com/Summwer/pro-pnj-bkz/tree/merge-enumbs-and-practical-cost-model)
# (dist,dist_param) -> n -> a strategy
# We restrict ourself to the EnumBS strategy predictor
strats_kyber = {
    ("binomial",3): {
        125: [(70,7,1), (74,8,1)],
        140: [ (74,7,1), (79,8,1), (81,9,1), (81,8,1) ],
        150: [ (58,8,1), (68,8,1), (74,9,1), (81,10,1), (102,10,1) ],
        160: [ (66,8,1), (68,9,2), (74,9,1), (81,10,2), (102,10,2) ],
        170: [ (58,8,1), (74,9,1), (79,9,1), (81,10,5), (102,10,3) ]
    }
}