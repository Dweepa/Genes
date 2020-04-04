import React, { useState, useEffect } from 'react';
import { Button } from 'reactstrap';
import "bootstrap/dist/css/bootstrap.min.css";
import './App.css';
import ReactDOM from "react-dom";
import posed from 'react-pose';
import { fadeIn } from 'react-animations'
import { CircularProgressbar } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

const percentage = 66;

<CircularProgressbar
value={percentage}
  text={`${percentage}%`}
  styles={{
    // Customize the root svg element
    root: {},
    // Customize the path, i.e. the "completed progress"
    path: {
      // Path color
      stroke: `rgba(62, 152, 199, ${percentage / 100})`,
      // Whether to use rounded or flat corners on the ends - can use 'butt' or 'round'
      strokeLinecap: 'butt',
      // Customize transition animation
      transition: 'stroke-dashoffset 0.5s ease 0s',
      // Rotate the path
      transform: 'rotate(0.25turn)',
      transformOrigin: 'center center',
    },
    // Customize the circle behind the path, i.e. the "total progress"
    trail: {
      // Trail color
      stroke: '#d6d6d6',
      // Whether to use rounded or flat corners on the ends - can use 'butt' or 'round'
      strokeLinecap: 'butt',
      // Rotate the trail
      transform: 'rotate(0.25turn)',
      transformOrigin: 'center center',
    },
    // Customize the text
    text: {
      // Text color
      fill: '#f88',
      // Text size
      fontSize: '16px',
    },
    // Customize background - only used when the `background` prop is true
    background: {
      fill: '#3e98c7',
    },
  }}
/>;
