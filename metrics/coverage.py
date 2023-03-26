#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid


class Coverage:
    def __init__(self):
        rospy.init_node('coverage_node', anonymous=True)
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, callback=self.coverage_callback, queue_size=1)
        self.TOTAL_AREA: float = 100
        self.OBSTACLE_THR: int = 65
        self.FREE_THR: int = 25
        self.covered_area: int = 0
        self.coverage_rate: float = 0
        rospy.spin()

    def coverage_callback(self, msg: OccupancyGrid):
        map_data = msg.data
        res = msg.info.resolution
        cell_counts = sum([1 for i in map_data if 0 <= i < self.FREE_THR])
        self.covered_area = res * res * cell_counts
        self.coverage_rate = self.covered_area / self.TOTAL_AREA
        rospy.loginfo_throttle(1, f'Your current coverage rate is %.1f%%', self.coverage_rate * 100)


if __name__ == '__main__':
    try:
        coverage_node = Coverage()
    except rospy.ROSInterruptException:
        pass
    finally:
        rospy.loginfo(f'Your final coverage rate is %.1f%%', coverage_node.coverage_rate * 100)

