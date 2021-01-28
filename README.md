# nfl-win-probability
A win probability model to answer Dylan + Adrian's question about timeouts. Fit 10 bootstrap MLPs to predict win probability for a given game state. These situational win probabilities can then be turned around to answer Dylan + Adrian's original question about whether the timeout is better than the delay-of-game.

Here's the dashboard with win probabilities and advice on the delay/timeout conundrum: https://joe-murphy.shinyapps.io/NFLWP/

A couple notes to self:
  - Situations with $\leq$ 3 minutes remaining and |Score Differential|$<10$ are upsampled (with replacement) s.t. the number of upsample rows added equals $\approx .2$ the number of rows in the original dataset. In so doing, I hoped to encourage (hopefully) better gradient approximations late in games -- as shown by Clauset et al. (https://arxiv.org/pdf/1503.03509.pdf), the arcsine law offers a solid approximation lead change times in the NFL, and in particular, the increased probability of late game lead changes. Notably, my cutoff for upsampling is in no way rigorous: the three minute bound is meant to serve as a loose bound on the time at which the arcsine function begins its ascent, while the score differential bound is meant to filter non-competitive scenarios. 
  - The MLPs have no bells and whistles -- no regularization/skip-layer/batch-norm or anything like that. Tensorflow installation was messier than I had time for, so any legitimate form of this model should probably be kicked over there eventually. Also, since it was run locally, I only fit 10 models -- in a more serious effort, this number ought to be higher. 
  - Obviously dumb situations (e.g. 14-14 with 60:00 game minutes remaining) will break the model.
