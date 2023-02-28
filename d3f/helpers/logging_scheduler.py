
import time

class LoggingScheduler():

    def __init__(self):
        
        self.start_time = self.get_current_time()
        self.last_log_time = self.get_current_time()

        self.last_step_number = None

        self.elapsed_time_since_start = 0.0
        self.elapsed_time_since_last_log = 0.0
        
    def update_with_step_number(self, global_step_number):

        if self.has_step_number_changed(global_step_number):
            
            if self.has_enough_time_elapsed_since_last_log():
                
                self.last_log_time = self.get_current_time()
                self.log_this_step = True

            else:

                self.log_this_step = False

    def should_we_log_this_step(self):
        return self.log_this_step

    def has_step_number_changed(self,global_step_number):
        changed = global_step_number != self.last_step_number
        self.last_step_number = global_step_number
        return changed
            
    def has_enough_time_elapsed_since_last_log(self):
        self.update_elapsed_times()

        seconds = 1
        minutes = 60
        hours = 3600

        time_between_logs = 1*hours

        if self.elapsed_time_since_start < 1*minutes:
            time_between_logs = 10*seconds
        elif self.elapsed_time_since_start < 15*minutes:
            time_between_logs = 1*minutes
        elif self.elapsed_time_since_start < 2*hours:
            time_between_logs = 10*minutes

        return self.elapsed_time_since_last_log > time_between_logs

    def update_elapsed_times(self):
        current_time = self.get_current_time()
        self.elapsed_time_since_start = current_time - self.start_time
        self.elapsed_time_since_last_log = current_time - self.last_log_time

    def get_current_time(self):
        return time.time()