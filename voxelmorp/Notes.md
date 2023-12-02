- Validation data: Gacr_01_006_01_580_m -> for evalulation of all methods
- Select first frame as fixed frame and transfomr the remain moving frames to fixed frame => moved_frames. 

```python

# model: voxelmorp, spam, airlab, labreg

fix_frame = video[0]
se = list()
for frame in video[1:]:
    moved_frame = model.predict(fix_frame, moving_frame)
    loss = (fix_frame - moved_frame)**2
    se.append(loss)

mse = np.mean(se)
```



## Basic data split
- Data: training data + validation data
- Train model, use training data
- Evaluation: validation data

## Advanced data split 
- When: machine learning benchmark competition
- Data: training + validation + test
- Provide training and validation data to teams
- Keep test to decide the winner



## Virutal environment
- Isolated environment that you can install specific packages and library 
- 