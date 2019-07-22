from unit_analysis import *



firecares_id_list = ['97477','81147','88539','77989','98606',
 '91106','93345','75500','79592','94264','99082',
 '77936','81154','77863','78827','77656','93717','74731','100262','90552']


for firecares_id in firecares_id_list:
    try:
        dep = unit_analysis(firecares_id)
        dep.apparatus_query()
        dep.first_due_analysis()
    except:
        print("Error on " + firecares_id)
