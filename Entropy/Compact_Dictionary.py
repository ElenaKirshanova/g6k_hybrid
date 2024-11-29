from Entropy.Largelog import largelog

class par_comp_dic:
    def __init__(self, offset, dic, count, p):
        self.offset = offset
        self.dic = dic
        self.comp_dic = dic
        self.len = len(dic)
        self.count = count
        self.p = p

    def __repr__(self):
        if self.p==1:
            return("List with all " + str(self.len) + " different weight distributions")
        if self.offset<0:
            return("List with " + str(self.len) + " different weight distributions, each having probability at least 2^(-H(.)n+" + str(-self.offset) + "), total probability is " + str("{:.3f}".format(float(self.p))))
        elif self.offset==0:
            return("List with " + str(self.len) + " different weight distributions, each having probability at least 2^(-H(.)n), total probability is " + str("{:.3f}".format(float(self.p))))
        else:
            return("List with " + str(self.len) + " different weight distributions, each having probability at least 2^(-H(.)n-" + str(self.offset) + "), total probability is " + str("{:.3f}".format(float(self.p))))

    def logcount(self, base=2):
        return largelog(self.count, base=base)

empty_comp_dic=par_comp_dic(-float('inf'),[],0,0)
