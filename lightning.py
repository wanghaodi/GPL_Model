#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/4/21 6:52 下午
# @Author  : Haodi Wang
# @FileName: lightning.py
# @Software: PyCharm
# @contact: whdi@foxmail.com
#           whd@seu.edu.cn

def nice_print(msg, last=False):
    print()
    print("\033[0;35m" + msg + "\033[0m")
    if last:
        print()


lightning_logo = """
                    ####
                ###########
             ####################
         ############################
    #####################################
##############################################
#########################  ###################
#######################    ###################
####################      ####################
##################       #####################
################        ######################
#####################        #################
######################     ###################
#####################    #####################
####################   #######################
###################  #########################
##############################################
    #####################################
         ############################
             ####################
                  ##########
                     ####
"""