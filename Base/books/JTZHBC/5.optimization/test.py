#coding:utf-8

import optimization

#s = [1,4,3,2,7,3,6,3,2,4,5,3]

#print optimization.printschedule(s)

#print optimization.schedulecost(s)

domain = [(0,9)]*len(optimization.people)*2
#s = optimization.randomoptimize(domain,optimization.schedulecost)

#s = optimization.hillclimb(domain,optimization.schedulecost)

#s = optimization.annealingoptimize(domain,optimization.schedulecost)

s = optimization.geneticoptimize(domain,optimization.schedulecost)

print optimization.schedulecost(s)
print optimization.printschedule(s)


'''
这两种及大多数优化方法都假设：大多数问题，最优解应该接近于其他的最优解。
但某些特殊情况不一定有效。比如存在陡峭的突变的最优解。
'''
