import rospy
from std_srvs.srv import Empty, EmptyResponse
import numpy as np

from do_eval import eval_main, construct_lfd_env, END_LFD

class lfd_server_node(object):
    def __init__(self) -> None:
        self.init_env()
        print('LFD Server Node Initialized')

        rospy.Service('/lfd_start', Empty, self.start_lfd)
        rospy.Service('/lfd_end', Empty, self.end_lfd)
        print('LFD Server Node Services advertised')

        rospy.spin()

    def start_lfd(self, req):
        print('Start LFD')
        END_LFD[0] = False
        eval_main(with_planning=True, env=self.env)
        return EmptyResponse()
    
    def end_lfd(self, req):
        print('End LFD')
        END_LFD[0] = True
        return EmptyResponse()
    
    def init_env(self):
        self.env = construct_lfd_env(setup_robots = True)
    


if __name__ == '__main__':
    lfd_server_node()